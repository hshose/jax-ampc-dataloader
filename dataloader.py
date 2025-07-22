import jax
import jax.numpy as jnp
import numpy as np

import threading
from concurrent.futures import ThreadPoolExecutor
import queue
import collections, itertools

import time 

class GpuDataLoader:
    def __init__(self, dataset, rng_key, batch_size=1, shuffle=False):
        print("Putting dataset on GPU:")
        # "WTF?! WHY IS device_put 2x faster than jnp.array()?"
        start_time = time.time()
        # self.X = jax.device_put(dataset.X)
        self.X = jnp.asarray(dataset.X)
        # self.X = jnp.array(dataset.X)
        # self.U = jax.device_put(dataset.U)
        self.U = jnp.asarray(dataset.U)
        # self.U = jnp.array(dataset.U)
        # self.Y = jax.device_put(dataset.Y)
        self.Y = jnp.asarray(dataset.Y)
        # self.Y = jnp.array(dataset.Y)
        self.X.block_until_ready(),self.U.block_until_ready(),self.Y.block_until_ready()
        end_time = time.time()
        duration = end_time - start_time
        total_bytes = (
            dataset.X.nbytes +
            dataset.U.nbytes +
            dataset.Y.nbytes
        )
        total_gb = total_bytes / (1024**3)
        bandwidth = total_gb / duration
        print(f"Transferring {total_gb:.2f} GB took {duration:.2f} seconds, effective bandwidth: {bandwidth:.2f} GB/s")
        self.n_samples = dataset.X.shape[0]
        self.rng_key = rng_key
        self.batch_size = batch_size
        self.gpu_transfer_time = duration
        self.shuffle = shuffle
    
    def __len__(self):
        return int(np.ceil(self.n_samples / self.batch_size))
    
    def new_epoch(self):
        if self.shuffle:
            self.rng_key, subkey = jax.random.split(self.rng_key)
            self.indices = jax.random.permutation(subkey, self.n_samples)
        else:
            self.indices = jnp.arange(self.n_samples)
        self.idx_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx_idx >= self.n_samples:
            raise StopIteration
        end = min(self.idx_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[self.idx_idx:end]
        x_batch = self.X[batch_indices]
        y_batch = self.Y[batch_indices]
        u_batch = self.U[batch_indices]
        self.idx_idx = end
        return x_batch, u_batch, y_batch

class PrefetchToDevice:
    def __init__(self, loader, prefetch_size=4):
        self.loader = loader
        self.prefetch_size = prefetch_size

    def __len__(self):
        return len(self.loader)

    def new_epoch(self):
        if hasattr(self.loader, "new_epoch"):
            self.loader.new_epoch()

    def __iter__(self):
        self.iter = self._prefetch_iter()
        return self

    def __next__(self):
        return next(self.iter)

    def _prefetch_iter(self):
        queue = collections.deque()
        iterable = iter(self.loader)

        def enqueue(n):
            for data in itertools.islice(iterable, n):
                queue.append(jax.device_put(data))

        enqueue(self.prefetch_size)
        while queue:
            yield queue.popleft()
            enqueue(1)

class CpuDataLoader:
    def __init__(self, dataset, rng_key, batch_size=1, host_prefetch_size=1, num_workers=10, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.n_samples = dataset.X.shape[0]
        self.rng_key = rng_key
        
    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def new_epoch(self):
        if self.shuffle:
            self.rng_key, key = jax.random.split(self.rng_key)
            self.indices = np.array(jax.random.permutation(key, len(self.dataset)))
        else:
            self.indices = np.arange(len(self.dataset))
        self.idx_idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx_idx >= self.n_samples:
            raise StopIteration
        end = min(self.idx_idx+self.batch_size, self.n_samples)
        batch_indices = self.indices[self.idx_idx:end]
        X_batch = self.dataset.X[batch_indices]
        U_batch = self.dataset.U[batch_indices]
        Y_batch = self.dataset.Y[batch_indices]
        self.idx_idx = end
        return X_batch, U_batch, Y_batch

class AsyncCpuDataLoader:
    def __init__(self, dataset, rng_key, batch_size=1, host_prefetch_size=1, num_workers=10, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        # self.prefetch_size = prefetch_size
        self.host_prefetch_size = host_prefetch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))
        self.rng_key = rng_key
        

    def __len__(self):
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def _shuffle_indices(self):
        if self.shuffle:
            self.rng_key, key = jax.random.split(self.rng_key)
            idx = np.array(jax.random.permutation(key, len(self.dataset)))
        else:
            idx = np.arange(len(self.dataset))
        n_full   = (len(idx) // self.batch_size) * self.batch_size         # drop leftovers
        self.indices = idx[:n_full].reshape(-1, self.batch_size)
    
    def _enqueue_batch(self, batch_indices):
        try:
            X_batch = self.dataset.X[batch_indices]
            U_batch = self.dataset.U[batch_indices]
            Y_batch = self.dataset.Y[batch_indices]
            self.q.put((X_batch, U_batch, Y_batch), block=True)
        except Exception as e:
            print("Worker failed while enqueuing a batch:", e, flush=True)
            raise

    def new_epoch(self):
        self._shuffle_indices()
        self.q = queue.Queue(maxsize=self.host_prefetch_size)
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)
        self.total_batches = len(self.indices)
        self._batch_count = 0
        for idx_vec in self.indices:
            self.executor.submit(self._enqueue_batch, idx_vec)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self._batch_count >= self.total_batches:
            self.executor.shutdown(wait=True)
            raise StopIteration
        batch = self.q.get()
        self._batch_count += 1
        return batch