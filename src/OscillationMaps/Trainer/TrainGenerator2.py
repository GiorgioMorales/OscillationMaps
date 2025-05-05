import os
from tqdm import trange
import torch.optim as optim
from OscillationMaps.utils import *
from OscillationMaps.Models.unet import *
from OscillationMaps.Models.model import *
from OscillationMaps.Trainer.loss import *
from OscillationMaps.Models.network import Network
from OscillationMaps.Data.DataLoader import HDF5Dataset

from OscillationMaps.Models.nn import GroupNorm32
torch.serialization.add_safe_globals([UNet])
torch.serialization.add_safe_globals([nn.Sequential, SiLU, EmbedSequential, EmbedBlock, nn.ModuleList,
                                      nn.Conv2d, nn.Module, Downsample, Upsample, nn.AvgPool2d, ResBlock,
                                      QKVAttentionLegacy, AttentionBlock, QKVAttention, GroupNorm32,
                                      nn.GroupNorm, nn.Identity, nn.Dropout, nn.Conv1d])


class TrainModel:
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

    def normalize(self, indices_train, batch_size):
        running_mean = None
        running_var = None
        n_samples = 0
        batches = np.array_split(indices_train, len(indices_train) // batch_size)

        for batch_indices in batches:
            batch = [self.dataset[idx] for idx in batch_indices]
            p_t_nu_batch = torch.stack(
                [b[0][:self.crop, :self.crop, :, :] for b in batch])

            batch_mean = p_t_nu_batch.mean(dim=(0, 1, 2, 3, 4))
            batch_var = p_t_nu_batch.var(dim=(0, 1, 2, 3, 4), unbiased=False)

            if running_mean is None:
                running_mean = batch_mean
                running_var = batch_var
            else:
                delta = batch_mean - running_mean
                new_n_samples = n_samples + p_t_nu_batch.numel()

                running_mean += delta * (p_t_nu_batch.numel() / new_n_samples)
                running_var += (batch_var * p_t_nu_batch.numel() + delta ** 2 * n_samples * p_t_nu_batch.numel() / new_n_samples) / new_n_samples

            n_samples += p_t_nu_batch.numel()

        # Final dataset mean and standard deviation
        self.dataset_mean = running_mean
        self.dataset_std = torch.sqrt(running_var)

        # Compute max differences AFTER normalization
        global_max_diff = None
        global_min_diff = None
        for batch_indices in batches:
            batch = [self.dataset[idx] for idx in batch_indices]
            p_t_nu_vacuum_batch = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in batch])

            # Normalize
            p_t_nu_vacuum_batch = (p_t_nu_vacuum_batch - self.dataset_mean) / self.dataset_std

            # Compute per-image per-channel difference
            per_image_diffs = (p_t_nu_vacuum_batch.amax(dim=(1, 2)) - p_t_nu_vacuum_batch.amin(dim=(1, 2))).view(len(batch), -1)

            # Find max and min differences across all images seen so far
            batch_max_diff = per_image_diffs.max()  # Max difference per channel across batch
            batch_min_diff = per_image_diffs.min()  # Min difference per channel across batch

            if global_max_diff is None:
                global_max_diff = batch_max_diff
                global_min_diff = batch_min_diff
            else:
                global_max_diff = torch.maximum(global_max_diff, batch_max_diff)
                global_min_diff = torch.minimum(global_min_diff, batch_min_diff)

        # Store final values
        self.dataset_max_diff = global_max_diff
        self.dataset_min_diff = global_min_diff

    def train(self, epochs=1001, filepath='', batch_size = 24):
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]
        self.model.set_loss(self.criterion)
        self.optimizer = optim.Adam(self.model.denoise_fn.parameters(), lr=5e-5)
        self.model.denoise_fn.train()

        # if self.verbose:
        #     print('Calculating statistics...')
        # self.normalize(indices_train, batch_size)
        # self.model.set_stats(max_diff=self.dataset_max_diff, min_diff=self.dataset_min_diff)
        if self.verbose:
            print('Training...')

        # self.model.denoise_fn = torch.load(filepath)
        for epoch in trange(epochs):
            loss_global = 0
            np.random.shuffle(indices_train)  # Shuffle the indices at the start of each epoch

            # Split indices into batches
            num_batches = len(indices_train) // batch_size
            batches = np.array_split(indices_train, num_batches)

            for batch_indices in batches:
                # Load batches as chunks
                batch = [self.dataset[idx] for idx in batch_indices]
                p_t_nu_batch = torch.stack([b[0][:self.crop, :self.crop, :, :] for b in batch])  # (batch_size, 120, 120, 3, 3)
                p_t_nu_vacuum_batch = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in batch]) # (batch_size, 120, 120, 3, 3)
                b, h, w, ch, d = p_t_nu_batch.shape
                p_t_nu_batch = p_t_nu_batch.permute(0, 3, 4, 1, 2).view(b, ch * d, h, w).to(self.device)
                p_t_nu_vacuum_batch = p_t_nu_vacuum_batch.permute(0, 3, 4, 1, 2).view(b, ch * d, h, w).to(self.device)

                # Normalize
                # p_t_nu_batch = (p_t_nu_batch - self.dataset_mean)  / self.dataset_std
                # p_t_nu_vacuum_batch = (p_t_nu_vacuum_batch - self.dataset_mean)  / self.dataset_std
                mean_p = p_t_nu_batch.mean(dim=(-1, -2), keepdim=True)
                std_p = p_t_nu_batch.std(dim=(-1, -2), keepdim=True)
                mean_v = p_t_nu_vacuum_batch.mean(dim=(-1, -2), keepdim=True)
                std_v = p_t_nu_vacuum_batch.std(dim=(-1, -2), keepdim=True)
                p_t_nu_batch = (p_t_nu_batch - mean_p) / std_p.clamp(min=1e-6)
                p_t_nu_vacuum_batch = (p_t_nu_vacuum_batch - mean_v) / std_v.clamp(min=1e-6)

                # min_p = p_t_nu_batch.min(dim=(-1, -2), keepdim=True).values
                # max_p = p_t_nu_batch.max(dim=(-1, -2), keepdim=True).values
                # min_v = p_t_nu_vacuum_batch.min(dim=(-1, -2), keepdim=True).values
                # max_v = p_t_nu_vacuum_batch.max(dim=(-1, -2), keepdim=True).values
                # p_t_nu_batch = (p_t_nu_batch - min_p) / (max_p - min_p).clamp(min=1e-6)
                # p_t_nu_vacuum_batch = (p_t_nu_vacuum_batch - min_v) / (max_v - min_v).clamp(min=1e-6)

                # for bb in range(len(batch)):
                #     for i in range(p_t_nu_batch.size(1)):
                #         p_t_nu_batch[bb, i, :, :] = ((p_t_nu_batch[bb, i, :, :] - p_t_nu_batch[bb, i, :, :].min())  /
                #                                      (p_t_nu_batch[bb, i, :, :].max() - p_t_nu_batch[bb, i, :, :].min()))
                #         p_t_nu_vacuum_batch[bb, i, :, :] = ((p_t_nu_vacuum_batch[bb, i, :, :] - p_t_nu_vacuum_batch[bb, i, :, :].min())  /
                #                                      (p_t_nu_vacuum_batch[bb, i, :, :].max() - p_t_nu_vacuum_batch[bb, i, :, :].min()))

                # Forward pass
                loss = self.model.forward(y_0=p_t_nu_batch, y_cond=p_t_nu_vacuum_batch)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_global += loss.item()

            torch.save(self.model.denoise_fn, filepath)

            if epoch % 1 == 0 and self.verbose:
                print('epoch:', epoch, 'loss:', loss_global / len(batches))

        # Analyze results
        self.model.denoise_fn = torch.load(filepath)
        osc_maps_gen = []
        indices_val = indices_train[0:2]
        val_data = [self.dataset[idx] for idx in indices_val]
        p_t_nu_val = torch.stack([b[0][:self.crop, :self.crop, :, :] for b in val_data]).to(self.device)
        p_t_nu_vacuum_val = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in val_data]).to(self.device)
        num_samples = p_t_nu_vacuum_val.shape[0]
        batch_size_val, total_size = 10, 0
        while total_size < num_samples:
            num_gen = num_samples - len(osc_maps_gen)
            for start_idx in range(0, num_gen, batch_size):
                end_idx = min(start_idx + batch_size, num_gen)
                b, h, w, ch, d = p_t_nu_vacuum_val[start_idx:end_idx, :, :, :, :].shape
                p_t_nu_vacuum_batch = p_t_nu_vacuum_val[start_idx:end_idx, :, :, :, :].permute(0, 3, 4, 1, 2).view(b, ch * d, h, w)
                # mean_v = p_t_nu_vacuum_batch.mean(dim=(-1, -2), keepdim=True)
                # std_v = p_t_nu_vacuum_batch.std(dim=(-1, -2), keepdim=True)
                # p_t_nu_vacuum_batch = (p_t_nu_vacuum_batch - mean_v) / std_v.clamp(min=1e-6)
                for bb in range(b):
                    for i in range(p_t_nu_vacuum_batch.size(1)):
                        p_t_nu_vacuum_batch[bb, i, :, :] = (
                                    (p_t_nu_vacuum_batch[bb, i, :, :] - p_t_nu_vacuum_batch[bb, i, :, :].min()) /
                                    (p_t_nu_vacuum_batch[bb, i, :, :].max() - p_t_nu_vacuum_batch[bb, i, :, :].min()))
                x_seq, _ = self.model.restoration(y_cond=p_t_nu_vacuum_batch.to(self.device))
                osc_maps_gen_batch = x_seq.cpu().detach().view(b, ch, d, h, w).permute(0, 3, 4, 1, 2).numpy()
                osc_maps_gen.append(osc_maps_gen_batch)
            total_size = len(np.vstack(osc_maps_gen))

        osc_maps_gen = np.vstack(osc_maps_gen)
        if self.plot:
            plot_osc_maps(osc_maps_gen[0, :, :, :, :], title='Osc. Maps w/ Matter effect')
            plot_osc_maps(p_t_nu_vacuum_val.cpu()[0, :, :, :, :], title='Osc. Maps in Vacuum')
            plot_osc_maps(p_t_nu_val.cpu()[0, :, :, :, :],)

    def train_ensemble(self, ensemble_size=1, epochs=1001):
        # If the folder does not exist, create it
        root = get_project_root()
        folder = os.path.join(root, "Models//saved_models//ModelType-" + str(self.complexity))
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
            self.model = self.reset_model()
            print("\tTraining ", mi + 1, "/", ensemble_size, " model")
            self.train(epochs=epochs, filepath=f)


if __name__ == '__main__':
    complexities = [2]
    for c in complexities:
        print("Training ensemble of models of complexity ", c)
        model = TrainModel(complexity=c, plotR=True, verbose=True)
        model.train_ensemble(ensemble_size=1, epochs=1000)
