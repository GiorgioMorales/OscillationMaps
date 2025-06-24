import os
import h5py
import numpy as np
from tqdm import trange


def load_h5_data(h5_file_path):
    data_dict = {}

    def recursive_read(h5_obj, prefix=""):
        for key in h5_obj.keys():
            full_key = f"{prefix}/{key}" if prefix else key
            item = h5_obj[key]
            if isinstance(item, h5py.Group):
                recursive_read(item, full_key)
            elif isinstance(item, h5py.Dataset):
                data_dict[full_key] = item[()]
    with h5py.File(h5_file_path, "r") as f:
        recursive_read(f)

    return data_dict


def create_dataset(extended=True):
    """
    Creates an HDF5 dataset with all oscillation maps, writing incrementally to save memory.
    :param extended: If True, creates a dataset using files from the ExtendedRange folder.
    """
    # Folder containing the HDF5 files
    if extended:
        data_folder = "/home/morales251/Documents/Projects/OscMaps-ExtendedRange"
        output_file = "/home/morales251/Documents/Projects/OscillationMaps/src/OscillationMaps/Data/Datasets/oscillation_maps_extended2.h5"
    else:
        data_folder = "Data/NormalRange"
        output_file = "/home/morales251/Documents/Projects/OscillationMaps/src/OscillationMaps/Data/Datasets/oscillation_maps2.h5"

    # Find all HDF5 files in the directory
    h5_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith(".h5")]
    num_files = len(h5_files)

    # Define dataset shapes
    shape_p_t_nu = (num_files, 120, 120, 3, 3)
    shape_p_t_nu_vacuum = (num_files, 120, 120, 3, 3)
    shape_U_PMNS = (num_files, 3, 3)
    shape_osc_par = (num_files, 6)
    shape_mass_square = (num_files, 3)

    # Create HDF5 file with chunked datasets
    with h5py.File(output_file, "w") as f_out:
        p_t_nu_dset = f_out.create_dataset("p_t_nu", shape=shape_p_t_nu, dtype=np.float32, chunks=(1, 120, 120, 3, 3))
        p_t_nu_vacuum_dset = f_out.create_dataset("p_t_nu_vacuum", shape=shape_p_t_nu_vacuum, dtype=np.float32, chunks=(1, 120, 120, 3, 3))
        U_PMNS_dset = f_out.create_dataset("U_PMNS", shape=shape_U_PMNS, dtype=np.complex64, chunks=(1, 3, 3))
        osc_par_dset = f_out.create_dataset("osc_par", shape=shape_osc_par, dtype=np.float32, chunks=(1, 6))
        mass_square_dset = f_out.create_dataset("mass_square", shape=shape_mass_square, dtype=np.float32, chunks=(1, 3))

        # Process each HDF5 file and write incrementally
        for i in trange(num_files):
            h5_file = h5_files[i]
            with h5py.File(h5_file, "r") as f:
                p_t_nu = np.zeros((120, 120, 3, 3), dtype=np.float32)
                p_t_nu_vacuum = np.zeros((120, 120, 3, 3), dtype=np.float32)

                # Read maps with and without matter interaction
                p_t_nu[:, :, 0, :] = f["p_t_nu_e"]
                p_t_nu[:, :, 1, :] = f["p_t_nu_mu"]
                p_t_nu[:, :, 2, :] = f["p_t_nu_tau"]
                p_t_nu_vacuum[:, :, 0, :] = f["p_t_nu_e_vacuum"]
                p_t_nu_vacuum[:, :, 1, :] = f["p_t_nu_mu_vacuum"]
                p_t_nu_vacuum[:, :, 2, :] = f["p_t_nu_tau_vacuum"]

                # print()
                # print(p_t_nu[:, :, 0, 1] == p_t_nu[:, :, 1, 0])
                # print(p_t_nu[:, :, 0, 2] == p_t_nu[:, :, 2, 0])
                # print(p_t_nu[:, :, 1, 2] == p_t_nu[:, :, 2, 1])

                # Write to HDF5 file incrementally
                p_t_nu_dset[i] = p_t_nu
                p_t_nu_vacuum_dset[i] = p_t_nu_vacuum
                U_PMNS_dset[i] = np.array(f["U_PMNS"][:], dtype=np.complex64)
                osc_par_dset[i] = np.array(f["osc_par"])
                mass_square_dset[i] = np.array(f["mass_square"])


if __name__ == '__main__':
    create_dataset(extended=True)
