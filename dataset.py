import jax
import jax.numpy as jnp
import numpy as np
import tqdm

class MemoryAmpcDataset:
    def __init__(self, X, U, Y):
        self.X = X
        self.U = U
        self.Y = Y
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.U[idx], self.Y[idx]


def train_test_split(full_dataset, split=0.8):
    N_total = len(full_dataset)
    size_train = int(split * N_total)

    indices = np.arange(N_total)
    # You can shuffle if you want: np.random.shuffle(indices)
    train_indices = indices[:size_train]
    eval_indices  = indices[size_train:]

    # Convert to new datasets by indexing into X, U, Y
    train_dataset = MemoryAmpcDataset(
        full_dataset.X[train_indices],
        full_dataset.U[train_indices],
        full_dataset.Y[train_indices]
    )

    eval_dataset = MemoryAmpcDataset(
        full_dataset.X[eval_indices],
        full_dataset.U[eval_indices],
        full_dataset.Y[eval_indices]
    )

    return train_dataset, eval_dataset