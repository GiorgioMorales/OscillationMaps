import os
import itertools

import numpy as np

from OscillationMaps.utils import *
from OscillationMaps.Models.MLPs import *
from OscillationMaps.Trainer.loss import *
from OscillationMaps.Data.DataLoader import HDF5Dataset


import pynvml


def get_available_gpus():
    return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]


def get_least_used_gpus(n=4):
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_free_mem = []

    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_free_mem.append((i, mem_info.free))

    # Sort GPUs by most free memory
    sorted_gpus = sorted(gpu_free_mem, key=lambda x: x[1], reverse=True)
    selected_gpu_ids = [gpu_id for gpu_id, _ in sorted_gpus]

    # Repeat GPUs if fewer than n are available
    repeated_gpu_ids = list(itertools.islice(itertools.cycle(selected_gpu_ids), n))

    pynvml.nvmlShutdown()
    return [torch.device(f"cuda:{gpu_id}") for gpu_id in repeated_gpu_ids]


class EvalModel:
    def __init__(self, datapath='oscillation_maps_extended', verbose=False, plotR=False):
        self.crop = 120
        self.n_models = 5
        self.dataset = HDF5Dataset(datapath)
        self.devices = get_least_used_gpus(n=4)
        self.models = self.reset_model()
        self.optimizer = [torch.optim.Adam(mdl.parameters(), lr=5e-5) for mdl in self.models]
        self.criterion = nn.MSELoss()
        self.verbose = verbose
        self.plot = plotR
        self.dataset_mean, self.dataset_std = [0]*self.n_models, [0]*self.n_models
        self.dataset_min_diff = None
        self.dataset_max_diff = None

        self.X, self.Y, self.Xval, self.Yval = None, None, None, None

    def reset_model(self):
        models = []
        for i in range(self.n_models):
            models.append(MLP4(input_features=9))
        return models

    def eval(self, filepath=''):
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_val = indices[int(len(indices) * 0.8):]
        self.optimizer = [torch.optim.Adam(mdl.parameters(), lr=5e-5) for mdl in self.models]
        self.devices = [torch.device(f'cpu') for _ in range(self.n_models)]

        batch_size = 1
        for mdl in range(self.n_models):
            self.models[mdl].to(self.devices[mdl])
            stats = torch.load(f'{filepath}_NNmodel{mdl}_norm_stats.pt', map_location=self.devices[mdl])
            self.dataset_mean[mdl] = stats['mean']
            self.dataset_std[mdl] = stats['std']
            self.models[mdl].load_state_dict(torch.load(f'{filepath}_NNmodel{mdl}.pt', map_location=self.devices[mdl]))
            self.models[mdl].eval()

        selected_channels = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        with torch.no_grad():
            val_loss_global = [0] * self.n_models
            val_num_batches = max(1, len(indices_val) // batch_size)
            val_batches = np.array_split(indices_val, val_num_batches)

            for batch_indices in val_batches:
                batch = [self.dataset[idx] for idx in batch_indices]
                p_t_nu_batch = torch.stack([b[0][:self.crop, :self.crop, :, :] for b in batch])
                p_t_nu_vacuum_batch = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in batch])
                osc_par_batch = torch.stack([b[3][[0, 1, 2, 3, 4, 5]] for b in batch])

                osc_pred_batch = p_t_nu_vacuum_batch[0, :, :, :, :].cpu().numpy() * 0
                for mdl in range(self.n_models):
                    device = self.devices[mdl]
                    ch_i, ch_j = selected_channels[mdl]
                    H, W = p_t_nu_vacuum_batch.shape[1:3]

                    input_vacuum = p_t_nu_vacuum_batch[0, :, :, ch_i, ch_j].reshape(-1, 1)
                    target_image = p_t_nu_batch[0, :, :, ch_i, ch_j]
                    target_i = target_image.reshape(-1).to(device)
                    # Create positional grid
                    x_coords = torch.arange(H).view(-1, 1).repeat(1, W).flatten()
                    y_coords = torch.arange(W).repeat(H)
                    pos_grid = torch.stack([x_coords, y_coords], dim=1)
                    # Repeat position and osc params for each pixel
                    pos_grid = pos_grid.repeat(1, 1).repeat_interleave(1, dim=0).repeat(1, 1)
                    pos_grid = pos_grid.expand(H * W, 2)
                    osc_rep = osc_par_batch.repeat(H * W, 1)

                    input_vector = torch.cat([pos_grid, osc_rep, input_vacuum], dim=1).to(device)
                    input_vector = (input_vector - self.dataset_mean[mdl]) / self.dataset_std[mdl]

                    output_i = self.models[mdl](input_vector)
                    osc_pred_batch[:, :, ch_i, ch_j] = np.reshape(output_i.cpu().numpy(), (H, W))
                    loss = self.criterion(output_i, target_i)
                    val_loss_global[mdl] += loss.item()

                if np.sum(osc_pred_batch[:, :, 0, 2]) == 0:
                    osc_pred_batch[:, :, 0, 2] = 1 - (osc_pred_batch[:, :, 0, 0] + osc_pred_batch[:, :, 0, 1])
                if np.sum(osc_pred_batch[:, :, 1, 2]) == 0:
                    osc_pred_batch[:, :, 1, 2] = 1 - (osc_pred_batch[:, :, 0, 1] + osc_pred_batch[:, :, 2, 2])
                if np.sum(osc_pred_batch[:, :, 1, 0]) == 0:
                    osc_pred_batch[:, :, 1, 0] = 1 - (osc_pred_batch[:, :, 1, 2] + osc_pred_batch[:, :, 1, 1])
                if np.sum(osc_pred_batch[:, :, 2, 0]) == 0:
                    osc_pred_batch[:, :, 2, 0] = 1 - (osc_pred_batch[:, :, 0, 0] + osc_pred_batch[:, :, 1, 0])
                if np.sum(osc_pred_batch[:, :, 2, 1]) == 0:
                    osc_pred_batch[:, :, 2, 1] = 1 - (osc_pred_batch[:, :, 0, 1] + osc_pred_batch[:, :, 1, 1])

                plot_osc_maps(p_t_nu_batch[0, :, :, :, :], title='Real Osc. Maps w/ Matter effect')
                plot_osc_maps(osc_pred_batch, title='Pred. Osc. Maps w/ Matter effect')
                plot_osc_maps(p_t_nu_vacuum_batch[0, :, :, :, :], title='Osc. Maps in Vacuum')

            val_loss_avg = [vl / len(val_batches) for vl in val_loss_global]
            print(f'val_loss: {[round(l / len(val_batches), 4) for l in val_loss_avg]}')

    def eval_ensemble(self, ensemble_size=1):
        for i, dev in enumerate(self.devices):
            print(f"Model {i} assigned to {dev} - {torch.cuda.get_device_name(dev)}")
        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root, "Models//saved_models//ModelType-MLP4")
        if not os.path.exists(os.path.join(root, "Models//saved_models//")):
            os.mkdir(os.path.join(root, "Models//saved_models//"))
        if not os.path.exists(folder):
            os.mkdir(folder)

        for mi in range(ensemble_size):
            filepath = folder + "//Model-"
            filepath = [filepath] * ensemble_size
            filepath[mi] = filepath[mi] + "-Instance" + str(mi) + '.pth'
            f = filepath[mi]
            # Train the model
            self.models = self.reset_model()
            print("\tTraining ", mi + 1, "/", ensemble_size, " model")
            self.eval(filepath=f)


if __name__ == '__main__':
    modell = EvalModel(plotR=True, verbose=True)
    modell.eval_ensemble(ensemble_size=1)
