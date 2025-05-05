import os

import matplotlib.pyplot as plt
import numpy as np
import ot
from tqdm import trange, tqdm
import torch.optim as optim
from OscillationMaps.utils import *
from scipy.interpolate import interp1d
from OscillationMaps.Models.unet import *
from OscillationMaps.Models.model import *
from OscillationMaps.Trainer.loss import *
from OscillationMaps.Models.network import Network
from OscillationMaps.Data.DataLoader import HDF5Dataset
from OscillationMaps.PRCurves.getPRCurves import getPRCurves

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def OT(D1, D2):
    """Compute Optimal Transport for each position of the image"""
    D1 = D1.reshape(D1.shape[0], D1.shape[1], D1.shape[2], D1.shape[3] * D1.shape[4])
    D2 = D2.reshape(D2.shape[0], D2.shape[1], D2.shape[2], D2.shape[3] * D2.shape[4])

    # Per-pixel comparison
    OTs = np.zeros((D1.shape[1], D1.shape[2]))
    for i in range(D1.shape[1]):
        for j in range(D1.shape[2]):
            X_np, Y_np = D1[:, i, j, :], D2[:, i, j, :]
            a = np.ones((X_np.shape[0],)) / X_np.shape[0]
            b = np.ones((Y_np.shape[0],)) / Y_np.shape[0]
            M = ot.dist(X_np, Y_np, metric='euclidean')
            OTs[i, j] = ot.sinkhorn2(a, b, M, reg=1e-1)
    return OTs



class EvaluateModel:
    def __init__(self, datapath='oscillation_maps_extended', modelType='DDPM', complexity=1, verbose=False, plotR=False):
        self.modelType = modelType
        self.crop = 64
        self.complexity = complexity
        self.dataset = HDF5Dataset(datapath)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.T = 100
        self.model = self.reset_model()
        self.model.denoise_fn.to(self.device)
        self.optimizer = optim.Adam(self.model.denoise_fn.parameters(), lr=5e-5)
        self.criterion = nn.MSELoss()
        self.verbose = verbose
        self.plot = plotR
        self.dataset_mean, self.dataset_std = 0, 1
        self.dataset_min_diff = None
        self.dataset_max_diff = None

    def reset_model(self):
        unet = UNet(
            image_size=self.crop,
            in_channel=18,
            inner_channel=32,
            out_channel=9,
            res_blocks=self.complexity,
            attn_res=[4]
        ).to(self.device)
        return Network(unet=unet, timesteps=self.T, device=self.device)


    def eval_ensemble(self, ensemble_size=10):
        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root, "Models//saved_models//ModelType-" + str(self.complexity))
        if not os.path.exists(os.path.join(root, "Models//saved_models//")):
            os.mkdir(os.path.join(root, "Models//saved_models//"))
        if not os.path.exists(folder):
            os.mkdir(folder)

        total_OTs = []
        for mi in range(ensemble_size):
            filepath = folder + "//Model-"
            filepath = [filepath] * ensemble_size
            filepath[mi] = filepath[mi] + "-Instance" + str(mi) + '.pth'

            # Configure data
            np.random.seed(7)
            indices = np.arange(len(self.dataset))
            np.random.shuffle(indices)
            indices_val = indices[int(len(indices) * 0.8):]

            # Compute PR curve
            self.model.denoise_fn = torch.load(filepath[mi])
            osc_maps_gen = []
            indices_val = indices_val
            val_data = [self.dataset[idx] for idx in indices_val][0:50]

            p_t_nu_vacuum_val = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in val_data]).to(self.device)
            num_samples = p_t_nu_vacuum_val.shape[0]
            batch_size_val, total_size = 50, 0
            pbar = tqdm(total=num_samples)
            while total_size < num_samples:
                num_gen = num_samples - len(osc_maps_gen)
                for start_idx in range(0, num_gen, batch_size_val):
                    end_idx = min(start_idx + batch_size_val, num_gen)
                    b, h, w, ch, d = p_t_nu_vacuum_val[start_idx:end_idx, :, :, :, :].shape
                    p_t_nu_vacuum_batch = p_t_nu_vacuum_val[start_idx:end_idx, :, :, :, :].permute(0, 3, 4, 1, 2).view(
                        b, ch * d, h, w)
                    mean_v = p_t_nu_vacuum_batch.mean(dim=(-1, -2), keepdim=True)
                    std_v = p_t_nu_vacuum_batch.std(dim=(-1, -2), keepdim=True)
                    p_t_nu_vacuum_batch = (p_t_nu_vacuum_batch - mean_v) / std_v.clamp(min=1e-6)
                    x_seq, _ = self.model.restoration(y_cond=p_t_nu_vacuum_batch.to(self.device))
                    osc_maps_gen_batch = x_seq.cpu().detach().view(b, ch, d, h, w).permute(0, 3, 4, 1, 2).numpy()
                    osc_maps_gen.append(osc_maps_gen_batch)
                    pbar.update(len(osc_maps_gen_batch))
                total_size = len(np.vstack(osc_maps_gen))

            if self.plot:
                plot_osc_maps(osc_maps_gen[0, :, :, :, :], title='Osc. Maps w/ Matter effect')
                plot_osc_maps(p_t_nu_vacuum_val.cpu()[0, :, :, :, :], title='Osc. Maps in Vacuum')
                # plot_osc_maps(p_t_nu_val.cpu()[0, :, :, :, :],)

            del p_t_nu_vacuum_batch, p_t_nu_vacuum_val

            # Concatenate all batches
            osc_maps_gen = np.vstack(osc_maps_gen)
            osc_maps_real = torch.stack([b[0][:self.crop, :self.crop, :, :] for b in val_data]).numpy()
            # Compute Optimal Transport
            OT_mat = OT(osc_maps_real, osc_maps_gen)
            total_OTs.append(OT_mat)
            plt.figure()
            plt.imshow(total_OTs)

        return total_OTs


if __name__ == '__main__':
    complexities = [2]
    for c in complexities:
        print("Evaluating ensemble of models of complexity ", c)
        model = EvaluateModel(complexity=c, plotR=True, verbose=True)
        model.eval_ensemble(ensemble_size=1)
