import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from OscillationMaps.utils import get_project_root


class HDF5Dataset(Dataset):
    def __init__(self, h5_file_path):
        """
        Custom PyTorch Dataset for loading data from an HDF5 file.
        :param h5_file_path: Path to the HDF5 file.
        """
        self.h5_file_path = 'Datasets/' + h5_file_path + '.h5'
        try:
            with h5py.File(self.h5_file_path, "r") as f:
                self.dataset_size = f["p_t_nu"].shape[0]  # Number of samples
        except FileNotFoundError:
            self.h5_file_path = os.path.join(get_project_root(), 'Data/', self.h5_file_path)
            with h5py.File(self.h5_file_path, "r") as f:
                self.dataset_size = f["p_t_nu"].shape[0]


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        """
        Fetch a single sample at the given index.
        """
        with h5py.File(self.h5_file_path, "r") as f:
            p_t_nu = f["p_t_nu"][index]
            p_t_nu_vacuum = f["p_t_nu_vacuum"][index]
            U_PMNS = f["U_PMNS"][index]
            osc_par = f["osc_par"][index]
            mass_square = f["mass_square"][index]

        # Convert to PyTorch tensors
        p_t_nu = torch.tensor(p_t_nu, dtype=torch.float32)
        p_t_nu_vacuum = torch.tensor(p_t_nu_vacuum, dtype=torch.float32)
        U_PMNS = torch.tensor(U_PMNS, dtype=torch.complex64)
        osc_par = torch.tensor(osc_par, dtype=torch.float32)
        mass_square = torch.tensor(mass_square, dtype=torch.float32)

        return p_t_nu, p_t_nu_vacuum, U_PMNS, osc_par, mass_square


if __name__ == '__main__':
    H5_file_path = "oscillation_maps_extended"
    dataset = HDF5Dataset(H5_file_path)

    indices = np.arange(len(dataset))
    batch_size = 16
    num_batches = len(indices) // batch_size
    batches = np.array_split(indices, num_batches)
    batch = [dataset[idx] for idx in batches[0]]
    p_t_nu_batch = torch.stack([b[0] for b in batch])
    p_t_nu_vacuum_batch = torch.stack([b[1] for b in batch])
    U_PMNS_batch = torch.stack([b[2] for b in batch])
    osc_par_batch = torch.stack([b[3] for b in batch])
    mass_square_batch = torch.stack([b[4] for b in batch])

    import matplotlib.pyplot as plt
    titles = [
        [r"$\nu_{e \to e}$", r"$\nu_{e \to \mu}$", r"$\nu_{e \to \tau}$"],
        [r"$\nu_{\mu \to e}$", r"$\nu_{\mu \to \mu}$", r"$\nu_{\mu \to \tau}$"],
        [r"$\nu_{\tau \to e}$", r"$\nu_{\tau \to \mu}$", r"$\nu_{\tau \to \tau}$"]
    ]
    fig, axes = plt.subplots(3, 3)
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            im = ax.imshow(p_t_nu_batch[0, :, :, i, j].detach().cpu().numpy(), cmap="viridis", aspect="auto")
            ax.set_title(titles[i][j], fontsize=14)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# Training loop
# for epoch in range(num_epochs):
#     np.random.shuffle(indices)  # Shuffle the indices at the start of each epoch
#
#     # Split indices into batches
#     num_batches = len(indices) // batch_size
#     batches = np.array_split(indices, num_batches)
#
#     for batch_indices in batches:
#         batch = [dataset[idx] for idx in batch_indices]  # Fetch batch samples
#
#         # Stack tensors
#         p_t_nu_batch = torch.stack([b[0] for b in batch])  # (batch_size, 120, 120, 3, 3)
#         p_t_nu_vacuum_batch = torch.stack([b[1] for b in batch])  # (batch_size, 120, 120, 3, 3)
#         U_PMNS_batch = torch.stack([b[2] for b in batch])  # (batch_size, 3, 3)
#
#         # Forward pass with your model (replace `model` with your actual model)
#         outputs = model(p_t_nu_batch)  # Example usage
#
#         # Compute loss, backpropagate, etc.
#         loss = loss_function(outputs, U_PMNS_batch)  # Example loss function
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#     print(f"Epoch {epoch + 1}/{num_epochs} completed.")
