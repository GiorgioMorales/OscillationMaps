import os
from OscillationMaps.utils import *
from OscillationMaps.Models.MLPs import *
from OscillationMaps.Trainer.loss import *
from OscillationMaps.Data.DataLoader import HDF5Dataset


class EvalModel:
    def __init__(self, datapath='oscillation_maps_extended', verbose=False, plotR=False):
        self.crop = 120
        self.model_is = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        self.n_models = len(self.model_is)
        self.simpler_models = [6]
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
        for i in self.model_is:
            if i in self.simpler_models:
                models.append(MLP4(input_features=8, sin=True))
            else:
                models.append(MLP4(input_features=9))
        return models

    def eval(self, filepath='', instance=0):
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_val = indices[int(len(indices) * 0.8):]
        self.optimizer = [torch.optim.Adam(mdl.parameters(), lr=5e-5) for mdl in self.models]
        self.devices = [torch.device(f'cpu') for _ in self.model_is]

        batch_size = 1
        for im, mdl in enumerate(self.model_is):
            if mdl in self.simpler_models:
                f = filepath + 'MLP4'
            else:
                f = filepath + 'MLP4'
            f = f + "//Model-" + "-Instance" + str(instance) + '.pth'
            self.models[im].to(self.devices[im])
            stats = torch.load(f'{f}_NNmodel{mdl}_norm_stats.pt', map_location=self.devices[im])
            self.dataset_mean[im] = stats['mean']
            self.dataset_std[im] = stats['std']
            self.models[im].load_state_dict(torch.load(f'{f}_NNmodel{mdl}.pt', map_location=self.devices[im]))
            self.models[im].eval()

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

                osc_pred_batch = p_t_nu_batch[0, :, :, :, :].cpu().numpy() * 0
                for im, mdl in enumerate(self.model_is):
                    device = self.devices[im]
                    ch_i, ch_j = selected_channels[mdl]
                    H, W = p_t_nu_batch.shape[1:3]

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

                    if mdl in self.simpler_models:
                        input_vector = torch.cat([pos_grid, osc_rep], dim=1).to(device)
                    else:
                        input_vector = torch.cat([pos_grid, osc_rep, input_vacuum], dim=1).to(device)
                    input_vector = (input_vector - self.dataset_mean[im]) / self.dataset_std[im]

                    output_i = self.models[im](input_vector)
                    osc_pred_batch[:, :, ch_i, ch_j] = np.reshape(output_i.cpu().numpy(), (H, W))
                    loss = self.criterion(output_i, target_i)
                    val_loss_global[im] += loss.item()

                if np.sum(osc_pred_batch[:, :, 0, 2]) == 0:
                    osc_pred_batch[:, :, 0, 2] = 1 - (osc_pred_batch[:, :, 0, 0] + osc_pred_batch[:, :, 0, 1])
                if np.sum(osc_pred_batch[:, :, 1, 0]) == 0:
                    osc_pred_batch[:, :, 1, 0] = 1 - (osc_pred_batch[:, :, 1, 2] + osc_pred_batch[:, :, 1, 1])
                if np.sum(osc_pred_batch[:, :, 1, 2]) == 0:
                    osc_pred_batch[:, :, 1, 2] = 1 - (osc_pred_batch[:, :, 1, 0] + osc_pred_batch[:, :, 1, 1])
                if np.sum(osc_pred_batch[:, :, 2, 0]) == 0:
                    osc_pred_batch[:, :, 2, 0] = 1 - (osc_pred_batch[:, :, 0, 0] + osc_pred_batch[:, :, 1, 0])
                if np.sum(osc_pred_batch[:, :, 2, 1]) == 0:
                    osc_pred_batch[:, :, 2, 1] = 1 - (osc_pred_batch[:, :, 0, 1] + osc_pred_batch[:, :, 1, 1])

                plot_osc_maps(p_t_nu_batch[0, :, :, :, :], title='Real Osc. Maps w/ Matter effect')
                plot_osc_maps(osc_pred_batch, title='Pred. Osc. Maps w/ Matter effect')
                # plot_osc_maps(p_t_nu_vacuum_batch[0, :, :, :, :], title='Osc. Maps in Vacuum')

            val_loss_avg = [vl / len(val_batches) for vl in val_loss_global]
            print(f'val_loss: {[round(l / len(val_batches), 4) for l in val_loss_avg]}')

    def eval_ensemble(self, ensemble_size=1):
        for i, dev in enumerate(self.devices):
            print(f"Model {i} assigned to {dev} - {torch.cuda.get_device_name(dev)}")
        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root, "Models//saved_models//ModelType-Sin-")

        for mi in range(ensemble_size):
            self.models = self.reset_model()
            print("\tTraining ", mi + 1, "/", ensemble_size, " model")
            self.eval(filepath=folder, instance=mi)


if __name__ == '__main__':
    modell = EvalModel(plotR=True, verbose=True)
    modell.eval_ensemble(ensemble_size=1)
