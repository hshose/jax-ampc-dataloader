import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import time
from pathlib import Path
import pathlib
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from tabulate import tabulate
import matplotlib.pyplot as plt

from dataset import train_test_split, MemoryAmpcDataset
from dataloader import AsyncCpuDataLoader, CpuDataLoader, GpuDataLoader, PrefetchToDevice


def test_dataloading(
    N_DATA      = 10_000_000,        # rows
    IN_DIM      = 10,
    OUT_DIM     = 50,
    BATCH_SIZE  = 100_000,
    NUM_LAYERS = 10,
    HIDDEN_DIM = 100,
    HOST_PREFETCH_SIZE = 100,
    HOST_PREFETCH_N_THREADS = 10,
    DEVICE_PREFETCH_SIZE = 4
    ):

    print(f"\n{'='*20} test_dataloading called with: N_DATA={N_DATA}, IN_DIM={IN_DIM}, OUT_DIM={OUT_DIM}, BATCH_SIZE={BATCH_SIZE}, NUM_LAYERS={NUM_LAYERS}, HIDDEN_DIM={HIDDEN_DIM} {'='*20}")

    rng = np.random.default_rng(0)
    X_np = rng.standard_normal((N_DATA, IN_DIM),  dtype=np.float64)
    Y_np = rng.standard_normal((N_DATA, OUT_DIM), dtype=np.float64)
    U_np = np.zeros((N_DATA, OUT_DIM), dtype=np.float64)   # placeholder “U”

    dataset_full = MemoryAmpcDataset(X_np, U_np, Y_np)

    keys = jax.random.split(jax.random.PRNGKey(42), NUM_LAYERS * 2)
    W_dummy = [jax.random.normal(keys[2*i], (HIDDEN_DIM, HIDDEN_DIM)) for i in range(NUM_LAYERS)]
    b_dummy = [jax.random.normal(keys[2*i+1], (HIDDEN_DIM,)) for i in range(NUM_LAYERS)]

    @jax.jit
    def train_step(X, Y):
        def loss(W,b,X,Y):
            w = jnp.ones((IN_DIM, HIDDEN_DIM), dtype=X.dtype)    # fake params
            h = X @ w
            for i in range(NUM_LAYERS):
                h = h @ W[i] + b[i]
                h = jax.nn.relu(h)
            w = jnp.ones((HIDDEN_DIM, OUT_DIM), dtype=X.dtype)
            y_pred = h @ w
            loss = jnp.mean((y_pred - Y) ** 2)
            return loss
        loss, grads = jax.value_and_grad(loss, argnums=(0,1))(W_dummy, b_dummy, X, Y)
        return loss, grads

    on_gpu_dataloader = GpuDataLoader(
        dataset_full,
        rng_key=jax.random.PRNGKey(42),
        batch_size=BATCH_SIZE,
        shuffle=True)
    
    duration_gpu_transfer_full_dataset = on_gpu_dataloader.gpu_transfer_time

    # warming up ;-)
    on_gpu_dataloader.new_epoch()
    for step, (X_, U_, Y_) in enumerate(on_gpu_dataloader):
        loss, grads = train_step(X_, Y_)            # async
    loss.block_until_ready()

    print(f"\nTesting pure GPU dataloading:")
    start_time = time.time()
    on_gpu_dataloader.new_epoch()
    for step, (X_, U_, Y_) in enumerate(tqdm.tqdm(on_gpu_dataloader, desc="train")):
        loss, grads = train_step(X_, Y_)            # async
    loss.block_until_ready()
    end_time = time.time()
    duration_gpu_dataloader = end_time - start_time
    print(f"GPU dataloader loop took {duration_gpu_dataloader:.2f} seconds")

    on_cpu_dataloader = CpuDataLoader(
        dataset_full,
        rng_key=jax.random.PRNGKey(42),
        batch_size=BATCH_SIZE,
        shuffle=True
        )

    print(f"\nTesting pure CPU dataloading:")
    start_time = time.time()
    on_cpu_dataloader.new_epoch()
    for step, (X_, U_, Y_) in enumerate(tqdm.tqdm(on_cpu_dataloader, desc="train")):
        X_,U_,Y_ = jax.device_put((X_,U_,Y_))
        loss, grads = train_step(X_, Y_)            # async
    loss.block_until_ready()
    end_time = time.time()
    duration_cpu_dataloader = end_time - start_time
    print(f"CPU dataloader loop took {end_time - start_time:.2f} seconds")

    print(f"\nTesting CPU dataloading with GPU prefetch:")
    on_cpu_prefetch_loader = PrefetchToDevice(on_cpu_dataloader, prefetch_size=DEVICE_PREFETCH_SIZE)
    start_time = time.time()
    on_cpu_prefetch_loader.new_epoch()
    for step, (X_, U_, Y_) in enumerate(
        tqdm.tqdm(
            # prefetch_to_device(on_cpu_dataloader, size=5),
            on_cpu_prefetch_loader,
        desc="train")):
        loss, grads = train_step(X_, Y_)            # async
    loss.block_until_ready()
    end_time = time.time()
    duration_cpu_with_gpu_prefetch_dataloader = end_time - start_time
    print(f"CPU dataloading with GPU prefetch loop took {duration_cpu_with_gpu_prefetch_dataloader:.2f} seconds")

    print(f"\nTesting CPU host prefetch and GPU prefetch:")
    # device_iter = prefetch_to_device(loader, size=4)
    prefetch_on_cpu_loader = AsyncCpuDataLoader(
        dataset      = dataset_full,
        rng_key      = jax.random.PRNGKey(42),
        batch_size   = BATCH_SIZE,
        shuffle      = True,
        num_workers = HOST_PREFETCH_N_THREADS,
        host_prefetch_size = HOST_PREFETCH_SIZE,      # keep 100 batches ahead on the host
    )
    prefetch_on_cpu_prefetch_loader = PrefetchToDevice(prefetch_on_cpu_loader, prefetch_size=DEVICE_PREFETCH_SIZE)
    start_time = time.time()
    prefetch_on_cpu_prefetch_loader.new_epoch()
    for step, (X_, U_, Y_) in enumerate(tqdm.tqdm(prefetch_on_cpu_prefetch_loader, desc="train")):
        loss, grads = train_step(X_, Y_)            # async
    loss.block_until_ready()
    end_time = time.time()
    duration_cpu_prefetch_with_gpu_prefetch_dataloader = end_time - start_time
    print(f"CPU host prefetch and GPU prefetch dataloader loop took {duration_cpu_prefetch_with_gpu_prefetch_dataloader:.2f} seconds")
    
    
    return duration_gpu_transfer_full_dataset, duration_gpu_dataloader, duration_cpu_dataloader, duration_cpu_with_gpu_prefetch_dataloader, duration_cpu_prefetch_with_gpu_prefetch_dataloader

if __name__=="__main__":

    ## DEFAULTS:    
    # N_DATA      = 10_000_000
    # IN_DIM      = 10
    # OUT_DIM     = 50
    # BATCH_SIZE  = 100_000
    # NUM_LAYERS = 10
    # HIDDEN_DIM = 100
    # HOST_PREFETCH_SIZE = 100
    # HOST_PREFETCH_N_THREADS = 10
    # DEVICE_PREFETCH_SIZE = 4
    
    batch_size_sweep = [10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000]
    hidden_dim_sweep = [50, 100, 200, 500, 1_000]
    n_data_sweep = [100_000, 200_000, 500_000, 1_000_000, 2_000_000, 5_000_000,  10_000_000]

    results_batch_size_sweep = []
    for batch_size in batch_size_sweep:
        results_batch_size_sweep += [test_dataloading(BATCH_SIZE=batch_size)]

    results_hidden_dim_sweep = []
    for hidden_dim in hidden_dim_sweep:
        results_hidden_dim_sweep += [test_dataloading(HIDDEN_DIM=hidden_dim)]

    results_n_data_sweep = []
    for n_data in n_data_sweep:
        results_n_data_sweep += [test_dataloading(N_DATA=n_data)]

    labels = ["dataset to_device", "GPU", "CPU", "CPU device prefetch", "CPU device and host prefetch"]

    def print_table(sweep_name, sweep_vals, results):
        print(f"\n{sweep_name} sweep results:")
        headers = [sweep_name] + labels
        table = []
        for val, res in zip(sweep_vals, results):
            table.append([val] + [f"{x:.2f}" for x in res])
        print(tabulate(table, headers=headers, tablefmt="github"))

    print_table("Batch Size", batch_size_sweep, results_batch_size_sweep)
    print_table("Hidden Dim", hidden_dim_sweep, results_hidden_dim_sweep)
    print_table("N Data", n_data_sweep, results_n_data_sweep)

    # Prepare data for plotting

    results_lists = [
        (batch_size_sweep, results_batch_size_sweep, "Batch Size"),
        (hidden_dim_sweep, results_hidden_dim_sweep, "Hidden Dim"),
        (n_data_sweep, results_n_data_sweep, "N Data"),
    ]

    # Flatten all results to find global min/max for y-axis
    all_times = np.concatenate([np.array(results).flatten() for _, results, _ in results_lists])
    y_min = 0.9*all_times.min()
    y_max = 1.1*all_times.max()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (xvals, results, xlabel) in zip(axes, results_lists):
        results = np.array(results)  # shape: (len(xvals), len(labels))
        for i, label in enumerate(labels):
            ax.plot(
                xvals,
                results[:, i],
                marker="o",
                label=label,
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Time (s)")
        ax.legend()
        ax.set_title(f"Sweep: {xlabel}")
        ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    output_path = Path(__file__).parent / "dataloadertest.pdf"
    plt.savefig(output_path, bbox_inches="tight")