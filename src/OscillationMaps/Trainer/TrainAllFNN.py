import os
import itertools
from tqdm import trange
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


class TrainModel:
    def __init__(self, datapath='oscillation_maps_extended', modelType='DDPM', complexity=1, verbose=False, plotR=False):
        self.modelType = modelType
        self.crop = 120
        self.complexity = complexity
        self.dataset = HDF5Dataset(datapath)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.devices = get_least_used_gpus(n=4)
        self.T = 100
        self.models = self.reset_model()
        self.optimizer = [torch.optim.Adam(mdl.parameters(), lr=5e-5) for mdl in self.models]
        self.criterion = nn.MSELoss()
        self.verbose = verbose
        self.plot = plotR
        self.dataset_mean, self.dataset_std = 0, 1
        self.dataset_min_diff = None
        self.dataset_max_diff = None

        self.X, self.Y, self.Xval, self.Yval = None, None, None, None

    def reset_model(self):
        models = []
        for i in range(4):
            models.append(MLP4(input_features=9))
        return models

    def normalize(self, indices_train, indices_val, path_stats):
        all_inputs = []
        all_targets = []

        selected_channels = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0)]

        for idx in indices_train:
            data = self.dataset[idx]
            p_t_nu = data[0][:self.crop, :self.crop, :, :]
            p_t_nu_vacuum = data[1][:self.crop, :self.crop, :, :]
            # osc_par = data[3][[0, 1, 2, 5]]
            osc_par = data[3][[0, 1, 2, 3, 4, 5]]
            H, W = p_t_nu_vacuum.shape[:2]

            # Create positional grid
            x_coords = torch.arange(H).view(-1, 1).repeat(1, W).flatten()
            y_coords = torch.arange(W).repeat(H)
            pos_grid = torch.stack([x_coords, y_coords], dim=1)

            # Repeat position and osc params for each pixel
            pos_grid = pos_grid.repeat(1, 1).repeat_interleave(1, dim=0).repeat(1, 1)
            pos_grid = pos_grid.expand(H * W, 2)
            osc_rep = osc_par.repeat(H * W, 1)

            input_vacuums, target_vectors = torch.zeros((len(pos_grid), 4)), torch.zeros((len(pos_grid), 6))
            for imdl in range(4):
                ch_i, ch_j = selected_channels[imdl]
                input_vacuum = p_t_nu_vacuum[:, :, ch_i, ch_j].reshape(-1, 1)
                target_image = p_t_nu[:, :, ch_i, ch_j]
                target_vector = target_image.reshape(-1)
                input_vacuums[:, imdl] = input_vacuum[:, 0]
                target_vectors[:, imdl] = target_vector
            for imdl in [4, 5]:
                ch_i, ch_j = selected_channels[imdl]
                target_image = p_t_nu[:, :, ch_i, ch_j]
                target_vector = target_image.reshape(-1)
                target_vectors[:, imdl] = target_vector

            input_vector = torch.cat([pos_grid, osc_rep, input_vacuums], dim=1)

            all_inputs.append(input_vector)
            all_targets.append(target_vectors)

        # Combine all data
        self.X = torch.cat(all_inputs, dim=0)
        # Compute normalization statistics
        self.dataset_mean = self.X.mean(dim=0)
        self.dataset_std = self.X.std(dim=0, unbiased=True)
        self.X = (self.X - self.dataset_mean) / self.dataset_std
        self.Y = torch.cat(all_targets, dim=0)

        torch.save({'mean': self.dataset_mean, 'std': self.dataset_std}, path_stats)

        all_inputs = []
        all_targets = []
        for idx in indices_val:
            data = self.dataset[idx]
            p_t_nu = data[0][:self.crop, :self.crop, :, :]
            p_t_nu_vacuum = data[1][:self.crop, :self.crop, :, :]
            osc_par = data[3][[0, 1, 2, 3, 4, 5]]
            H, W = p_t_nu_vacuum.shape[:2]

            # Create positional grid
            x_coords = torch.arange(H).view(-1, 1).repeat(1, W).flatten()
            y_coords = torch.arange(W).repeat(H)
            pos_grid = torch.stack([x_coords, y_coords], dim=1)

            # Repeat position and osc params for each pixel
            pos_grid = pos_grid.repeat(1, 1).repeat_interleave(1, dim=0).repeat(1, 1)
            pos_grid = pos_grid.expand(H * W, 2)
            osc_rep = osc_par.repeat(H * W, 1)

            input_vacuums, target_vectors = torch.zeros((len(pos_grid), 4)), torch.zeros((len(pos_grid), 6))
            for imdl in range(4):
                ch_i, ch_j = selected_channels[imdl]
                input_vacuum = p_t_nu_vacuum[:, :, ch_i, ch_j].reshape(-1, 1)
                target_image = p_t_nu[:, :, ch_i, ch_j]
                target_vector = target_image.reshape(-1)
                input_vacuums[:, imdl] = input_vacuum[:, 0]
                target_vectors[:, imdl] = target_vector
            for imdl in [4, 5]:
                ch_i, ch_j = selected_channels[imdl]
                target_image = p_t_nu[:, :, ch_i, ch_j]
                target_vector = target_image.reshape(-1)
                target_vectors[:, imdl] = target_vector

            input_vector = torch.cat([pos_grid, osc_rep, input_vacuums], dim=1)

            all_inputs.append(input_vector)
            all_targets.append(target_vectors)

        # Combine all data
        self.Xval = torch.cat(all_inputs, dim=0)
        self.Xval = (self.Xval - self.dataset_mean) / self.dataset_std
        self.Yval = torch.cat(all_targets, dim=0)

    def train(self, epochs=1001, filepath='', batch_size=128, lambda_5=0.5, lambda_6=0.5, lambda_c=0.5):
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        # indices = np.arange(1000)  # TODO
        np.random.shuffle(indices)
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]
        self.optimizer = [torch.optim.Adam(mdl.parameters(), lr=5e-5) for mdl in self.models]

        for i in range(4):
            self.models[i].to(self.devices[i])
            self.models[i].train()

        if self.verbose:
            print('Normalizing...')
        self.normalize(indices_train=indices_train, indices_val=indices_val, path_stats=f'{filepath}_NNmodel_all_norm_stats.pt')

        if self.verbose:
            print('Training...')

        for mdl in range(4):
            self.models[mdl].load_state_dict(torch.load(f'{filepath}_NNmodel{mdl}.pt', map_location=self.devices[mdl]))

        best_val_loss = [float('inf')] * 4
        for epoch in trange(epochs):
            self.models[0].train()
            self.models[1].train()
            self.models[2].train()
            self.models[3].train()
            loss_global = [0, 0, 0, 0]

            num_samples = self.X.size(0)
            batches = torch.randperm(num_samples).split(batch_size)
            ct = 0
            for batch_indices in batches:
                ct += 1
                # print(ct, "/", len(batches))
                x_batch = self.X[batch_indices]
                y_batch = self.Y[batch_indices]

                # ——— Forward pass for all four models ———————————————————————————————————————
                y1_pred = self.models[0](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].to(self.devices[0]))
                y2_pred = self.models[1](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]].to(self.devices[1]))
                y3_pred = self.models[2](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 10]].to(self.devices[2]))
                y4_pred = self.models[3](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 11]].to(self.devices[3]))

                Y1 = y_batch[:, 0].to(self.devices[0])
                Y2 = y_batch[:, 1].to(self.devices[1])
                Y3 = y_batch[:, 2].to(self.devices[2])
                Y4 = y_batch[:, 3].to(self.devices[3])

                # ——— Compute basic data losses (MSE) ————————————————————————————————————————
                loss_y1 = self.criterion(y1_pred.squeeze(), Y1)
                loss_y2 = self.criterion(y2_pred.squeeze(), Y2)
                loss_y3 = self.criterion(y3_pred.squeeze(), Y3)
                loss_y4 = self.criterion(y4_pred.squeeze(), Y4)

                # ——— Compute new‐target predictions and losses ——————————————————————————————
                y5_pred = y1_pred + y4_pred.to(self.devices[0]) - y3_pred.to(self.devices[0])
                y6_pred = y2_pred.to(self.devices[0]) + y4_pred.to(self.devices[0]) - y3_pred.to(self.devices[0])

                Y5 = y_batch[:, 4].to(self.devices[0])
                Y6 = y_batch[:, 5].to(self.devices[0])
                loss_y5 = self.criterion(y5_pred.squeeze(), Y5)
                loss_y6 = self.criterion(y6_pred.squeeze(), Y6)

                # ——— Compute soft constraint violations ————————————————————————————————————
                c1_violation = torch.relu(y1_pred + y4_pred.to(self.devices[0]) - 1.0)  # y1 + y4 <= 1
                c2_violation = torch.relu(y2_pred.to(self.devices[0]) + y4_pred.to(self.devices[0]) - 1.0)  # y2 + y4 <= 1
                penalty = torch.mean(c1_violation.pow(2) + c2_violation.pow(2))

                # ——— Compute loss and gradients ———————————————————————————————————————————
                # Compute each model’s combined scalar loss (only those terms in which that model participates).
                # Use torch.autograd.grad(...) to pull out just the gradients for that model’s parameters,
                # without touching the rest of the graph.
                # Manually copy those gradients into model.parameters() and then call optimizer.step()

                #  f1_loss = loss_y1  + λ5·loss_y5  + λc·mean(c1²)
                f1_loss = loss_y1 + lambda_5 * loss_y5 + lambda_c * torch.mean(c1_violation.pow(2))
                #  f2_loss = loss_y2  + λ6·loss_y6  + λc·mean(c2²)
                f2_loss = loss_y2.to(self.devices[0]) + lambda_6 * loss_y6 + lambda_c * torch.mean(c2_violation.pow(2))
                #  f3_loss = loss_y3  + λ5·loss_y5  + λ6·loss_y6
                f3_loss = loss_y3.to(self.devices[0]) + lambda_5 * loss_y5 + lambda_6 * loss_y6
                #  f4_loss = loss_y4  + λ5·loss_y5  + λ6·loss_y6  + λc·mean(c1² + c2²)
                f4_loss = loss_y4.to(self.devices[0]) + lambda_5 * loss_y5 + lambda_6 * loss_y6 + lambda_c * penalty

                # Model 1
                params1 = list(self.models[0].parameters())
                grads1 = torch.autograd.grad(f1_loss, params1, retain_graph=True, create_graph=False)
                for p, g in zip(params1, grads1):
                    p.grad = g
                self.optimizer[0].step()
                loss_global[0] += f1_loss.item()
                # Model 2
                params2 = list(self.models[1].parameters())
                grads2 = torch.autograd.grad(f2_loss, params2, retain_graph=True, create_graph=False)
                for p, g in zip(params2, grads2):
                    p.grad = g
                self.optimizer[1].step()
                loss_global[1] += f2_loss.item()
                # Model 3
                params3 = list(self.models[2].parameters())
                grads3 = torch.autograd.grad(f3_loss, params3, retain_graph=True, create_graph=False)
                for p, g in zip(params3, grads3):
                    p.grad = g
                self.optimizer[2].step()
                loss_global[2] += f3_loss.item()
                # Model 4
                params4 = list(self.models[3].parameters())
                grads4 = torch.autograd.grad(f4_loss, params4, retain_graph=False, create_graph=False)
                for p, g in zip(params4, grads4):
                    p.grad = g
                self.optimizer[3].step()
                loss_global[3] += f4_loss.item()

            # Validation
            self.models[0].eval()
            self.models[1].eval()
            self.models[2].eval()
            self.models[3].eval()

            with torch.no_grad():
                val_loss_global = [0, 0, 0, 0]

                num_samples = self.Xval.size(0)
                val_batches = torch.randperm(num_samples).split(batch_size)
                for batch_indices in val_batches:
                    x_batch = self.X[batch_indices]
                    y_batch = self.Y[batch_indices]

                    # ——— Forward pass for all four models ———————————————————————————————————————
                    y1_pred = self.models[0](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 8]].to(self.devices[0]))
                    y2_pred = self.models[1](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 9]].to(self.devices[1]))
                    y3_pred = self.models[2](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 10]].to(self.devices[2]))
                    y4_pred = self.models[3](x_batch[:, [0, 1, 2, 3, 4, 5, 6, 7, 11]].to(self.devices[3]))

                    Y1 = y_batch[:, 0].to(self.devices[0])
                    Y2 = y_batch[:, 1].to(self.devices[1])
                    Y3 = y_batch[:, 2].to(self.devices[2])
                    Y4 = y_batch[:, 3].to(self.devices[3])

                    # ——— Compute basic data losses (MSE) ————————————————————————————————————————
                    loss_y1 = self.criterion(y1_pred.squeeze(), Y1)
                    loss_y2 = self.criterion(y2_pred.squeeze(), Y2)
                    loss_y3 = self.criterion(y3_pred.squeeze(), Y3)
                    loss_y4 = self.criterion(y4_pred.squeeze(), Y4)

                    # ——— Compute new‐target predictions and losses ——————————————————————————————
                    y5_pred = y1_pred + y4_pred.to(self.devices[0]) - y3_pred.to(self.devices[0])
                    y6_pred = y2_pred.to(self.devices[0]) + y4_pred.to(self.devices[0]) - y3_pred.to(self.devices[0])

                    Y5 = y_batch[:, 4].to(self.devices[0])
                    Y6 = y_batch[:, 5].to(self.devices[0])
                    loss_y5 = self.criterion(y5_pred.squeeze(), Y5)
                    loss_y6 = self.criterion(y6_pred.squeeze(), Y6)

                    # ——— Compute soft constraint violations ————————————————————————————————————
                    c1_violation = torch.relu(y1_pred + y4_pred.to(self.devices[0]) - 1.0)  # y1 + y4 <= 1
                    c2_violation = torch.relu(y2_pred.to(self.devices[0]) + y4_pred.to(self.devices[0]) - 1.0)  # y2 + y4 <= 1
                    penalty = torch.mean(c1_violation.pow(2) + c2_violation.pow(2))

                    #  f1_loss = loss_y1  + λ5·loss_y5  + λc·mean(c1²)
                    f1_loss = loss_y1 + lambda_5 * loss_y5 + lambda_c * torch.mean(c1_violation.pow(2))
                    val_loss_global[0] += f1_loss.item()
                    #  f2_loss = loss_y2  + λ6·loss_y6  + λc·mean(c2²)
                    f2_loss = loss_y2.to(self.devices[0]) + lambda_6 * loss_y6 + lambda_c * torch.mean(c2_violation.pow(2))
                    val_loss_global[1] += f2_loss.item()
                    #  f3_loss = loss_y3  + λ5·loss_y5  + λ6·loss_y6
                    f3_loss = loss_y3.to(self.devices[0]) + lambda_5 * loss_y5 + lambda_6 * loss_y6
                    val_loss_global[2] += f3_loss.item()
                    #  f4_loss = loss_y4  + λ5·loss_y5  + λ6·loss_y6  + λc·mean(c1² + c2²)
                    f4_loss = loss_y4.to(self.devices[0]) + lambda_5 * loss_y5 + lambda_6 * loss_y6 + lambda_c * penalty
                    val_loss_global[3] += f4_loss.item()

                val_loss_avg = [vl / len(val_batches) for vl in val_loss_global]

                for mdl in range(4):
                    if val_loss_avg[mdl] <= best_val_loss[mdl]:
                        best_val_loss[mdl] = val_loss_avg[mdl]
                        torch.save(self.models[mdl].state_dict(), f'{filepath}_NNmodel{mdl}.pt')
                        if self.verbose:
                            print(f'>> Saved model {mdl} at epoch {epoch} with val loss {val_loss_avg[mdl]:.8f}')

            if epoch % 1 == 0 and self.verbose:
                print(f'epoch: {epoch} | '
                      f'train_loss: {[round(l / len(batches), 8) for l in loss_global]} | '
                      f'val_loss: {[round(l / len(val_batches), 8) for l in val_loss_global]}')

    def train_ensemble(self, ensemble_size=1, epochs=1001, batch_size=128):
        for i, dev in enumerate(self.devices):
            print(f"Model {i} assigned to {dev} - {torch.cuda.get_device_name(dev)}")
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
            self.models = self.reset_model()
            print("\tTraining ", mi + 1, "/", ensemble_size, " model")
            self.train(epochs=epochs, filepath=f, batch_size=batch_size)


if __name__ == '__main__':
    complexities = [3]
    for c in complexities:
        print("Training ensemble of models of complexity ", c)
        modell = TrainModel(complexity=c, plotR=True, verbose=True)
        modell.train_ensemble(ensemble_size=1, epochs=1000, batch_size=1024)
