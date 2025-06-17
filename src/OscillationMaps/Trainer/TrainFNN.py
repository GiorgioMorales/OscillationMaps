import os
import pynvml
import itertools
from tqdm import trange
from OscillationMaps.utils import *
from OscillationMaps.Models.MLPs import *
from OscillationMaps.Trainer.loss import *
from OscillationMaps.Data.DataLoader import HDF5Dataset


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


class TrainModel:
    def __init__(self, datapath='oscillation_maps_extended', verbose=False, plotR=False):
        self.crop = 120
        self.dataset = HDF5Dataset(datapath)
        self.devices = get_least_used_gpus(n=1)
        self.models = self.reset_model()
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=5e-5)
        self.criterion = nn.MSELoss()
        self.verbose = verbose
        self.plot = plotR
        self.dataset_mean, self.dataset_std = 0, 1
        self.dataset_min_diff = None
        self.dataset_max_diff = None

        self.X, self.Y, self.Xval, self.Yval = None, None, None, None

    def reset_model(self):
        return MLP4(input_features=9)

    def load_data(self, indices, model_i):
        all_inputs = []
        all_targets = []
        selected_channels = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

        for idx in indices:
            data = self.dataset[idx]
            p_t_nu = data[0][:self.crop, :self.crop, :, :]
            p_t_nu_vacuum = data[1][:self.crop, :self.crop, :, :]
            osc_par = data[3][[0, 1, 2, 3, 4, 5]]

            ch_i, ch_j = selected_channels[model_i]
            H, W = p_t_nu_vacuum.shape[:2]

            input_vacuum = p_t_nu_vacuum[:, :, ch_i, ch_j].reshape(-1, 1)
            target_image = p_t_nu[:, :, ch_i, ch_j]
            target_vector = target_image.reshape(-1)

            # Create positional grid
            x_coords = torch.arange(H).view(-1, 1).repeat(1, W).flatten()
            y_coords = torch.arange(W).repeat(H)
            pos_grid = torch.stack([x_coords, y_coords], dim=1)

            # Repeat position and osc params for each pixel
            pos_grid = pos_grid.repeat(1, 1).repeat_interleave(1, dim=0).repeat(1, 1)
            pos_grid = pos_grid.expand(H * W, 2)
            osc_rep = osc_par.repeat(H * W, 1)

            input_vector = torch.cat([pos_grid, osc_rep, input_vacuum], dim=1)

            all_inputs.append(input_vector)
            all_targets.append(target_vector)

        return all_inputs, all_targets

    def normalize(self, indices_train, indices_val, model_i, path_stats):
        # Read and combine training data
        all_inputs, all_targets = self.load_data(indices=indices_train, model_i=model_i)
        self.X = torch.cat(all_inputs, dim=0).to(self.devices[0])
        self.dataset_mean = self.X.mean(dim=0)
        self.dataset_std = self.X.std(dim=0, unbiased=True)
        self.X = (self.X - self.dataset_mean) / self.dataset_std
        self.Y = torch.cat(all_targets, dim=0).to(self.devices[0])

        torch.save({'mean': self.dataset_mean, 'std': self.dataset_std}, path_stats)

        # Read and combine validation data
        all_inputs, all_targets = self.load_data(indices=indices_val, model_i=model_i)
        self.Xval = torch.cat(all_inputs, dim=0).to(self.devices[0])
        self.Xval = (self.Xval - self.dataset_mean) / self.dataset_std
        self.Yval = torch.cat(all_targets, dim=0).to(self.devices[0])

    def train(self, epochs=1001, filepath='', batch_size=128):
        model_i = 4
        print("Model, ", str(model_i))
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]
        self.optimizer = torch.optim.Adam(self.models.parameters(), lr=5e-5)
        self.models.to(self.devices[0])
        self.models.train()

        if self.verbose:
            print('Normalizing...')
        self.normalize(indices_train=indices_train, indices_val=indices_val,
                       model_i=model_i, path_stats=f'{filepath}_NNmodel{model_i}_norm_stats.pt')
        if self.verbose:
            print('Training...')

        # for mdl in [model_i]:
        #     self.models[mdl].load_state_dict(torch.load(f'{filepath}_NNmodel{mdl}.pt'))

        best_val_loss = [float('inf')] * 4
        for epoch in trange(epochs):
            self.models.train()
            loss_global = 0

            num_samples = self.X.size(0)
            batches = torch.randperm(num_samples).split(batch_size)
            for batch_indices in batches:
                input_vector = self.X[batch_indices].to(self.devices[0])
                target_flat = self.Y[batch_indices].to(self.devices[0])
                output = self.models(input_vector)
                loss = self.criterion(output.squeeze(), target_flat)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_global += loss.item()

            # Validation
            self.models.eval()
            with torch.no_grad():
                val_loss_global = 0
                num_samples = self.Xval.size(0)
                val_batches = torch.randperm(num_samples).split(batch_size)
                for batch_indices in val_batches:
                    input_vector = self.Xval[batch_indices].to(self.devices[0])
                    target_flat = self.Yval[batch_indices].to(self.devices[0])
                    output = self.models(input_vector)
                    loss = self.criterion(output.squeeze(), target_flat)
                    val_loss_global += loss.item()

                val_loss_avg = val_loss_global / len(val_batches)

                if val_loss_avg <= best_val_loss:
                    best_val_loss = val_loss_avg
                    torch.save(self.models.state_dict(), f'{filepath}_NNmodel{model_i}.pt')
                    if self.verbose:
                        print(f'>> Saved model at epoch {epoch} with val loss {val_loss_avg:.4f}')

            if epoch % 1 == 0 and self.verbose:
                print(f'epoch: {epoch} | '
                      f'train_loss: {round(loss_global / len(batches), 4)} | '
                      f'val_loss: {round(val_loss_global / len(val_batches), 4)}')

    def train_ensemble(self, ensemble_size=1, epochs=1001, batch_size=128):
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
            self.train(epochs=epochs, filepath=f, batch_size=batch_size)


if __name__ == '__main__':
    modell = TrainModel(plotR=True, verbose=True)
    modell.train_ensemble(ensemble_size=1, epochs=1000, batch_size=16)
