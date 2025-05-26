import os
from tqdm import trange
import torch.multiprocessing as mp
from OscillationMaps.utils import *
from OscillationMaps.Models.unet_osc import *
from OscillationMaps.Models.model import *
from OscillationMaps.Trainer.loss import *
from OscillationMaps.Models.network import Network
from OscillationMaps.Data.DataLoader import HDF5Dataset

# from OscillationMaps.Models.nn import GroupNorm32
# torch.serialization.add_safe_globals([UNet])
# torch.serialization.add_safe_globals([nn.Sequential, SiLU, EmbedSequential, EmbedBlock, nn.ModuleList,
#                                       nn.Conv2d, nn.Module, Downsample, Upsample, nn.AvgPool2d, ResBlock,
#                                       QKVAttentionLegacy, AttentionBlock, QKVAttention, GroupNorm32,
#                                       nn.GroupNorm, nn.Identity, nn.Dropout, nn.Conv1d])


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
    pynvml.nvmlShutdown()
    return [torch.device(f"cuda:{gpu_id}") for gpu_id, _ in sorted_gpus[:n]]


def train_single_model(model_id, model, device, dataset, crop, criterion, selected_channel, indices_train, indices_val, filepath, epochs, batch_size, verbose):
    torch.manual_seed(7 + model_id)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    best_val_loss = float('inf')

    for epoch in trange(epochs, desc=f"Model {model_id}", position=model_id):
        np.random.shuffle(indices_train)
        num_batches = len(indices_train) // batch_size
        batches = np.array_split(indices_train, num_batches)
        loss_global = 0

        for batch_indices in batches:
            batch = [dataset[idx] for idx in batch_indices]
            p_t_nu_batch = torch.stack([b[0][:crop, :crop, :, :] for b in batch])
            p_t_nu_vacuum_batch = torch.stack([b[1][:crop, :crop, :, :] for b in batch])
            osc_par_batch = torch.stack([b[3][[0, 1, 2, 5], ] for b in batch])

            input_im = p_t_nu_vacuum_batch[:, :, :, selected_channel[0], selected_channel[1]]
            input_im = input_im[:, None, :, :].to(device)
            target = p_t_nu_batch[:, :, :, selected_channel[0], selected_channel[1]]
            target = target[:, None, :, :].to(device)
            osc_par = osc_par_batch.to(device)

            output = model(x=input_im, osc_params=osc_par)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_global += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_batches = np.array_split(indices_val, max(1, len(indices_val) // batch_size))
            for batch_indices in val_batches:
                batch = [dataset[idx] for idx in batch_indices]
                p_t_nu_batch = torch.stack([b[0][:crop, :crop, :, :] for b in batch])
                p_t_nu_vacuum_batch = torch.stack([b[1][:crop, :crop, :, :] for b in batch])
                osc_par_batch = torch.stack([b[3][[0, 1, 2, 5]] for b in batch])

                input_im = p_t_nu_vacuum_batch[:, :, :, selected_channel[0], selected_channel[1]]
                input_im = input_im[:, None, :, :].to(device)
                target = p_t_nu_batch[:, :, :, selected_channel[0], selected_channel[1]]
                target = target[:, None, :, :].to(device)
                osc_par = osc_par_batch.to(device)

                output = model(x=input_im, osc_params=osc_par)
                loss = criterion(output, target)
                val_loss += loss.item()

            val_loss /= len(val_batches)
            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f'{filepath}_model{model_id}.pt')
                if verbose:
                    print(f'Model {model_id} saved at epoch {epoch} with val loss {val_loss:.4f}')

        model.train()


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

    def reset_model(self):
        models = []
        for i in range(4):
            unet = UNet(
                image_size=self.crop,
                in_channel=1,
                inner_channel=32,
                out_channel=1,
                res_blocks=self.complexity,
                attn_res=[2]
            )
            models.append(unet)
        return models

    def train_parallel(self, epochs=1001, filepath='', batch_size=128):
        from torch.multiprocessing import Process

        torch.multiprocessing.set_start_method('spawn', force=True)
        torch.multiprocessing.set_sharing_strategy("file_system")

        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]
        selected_channels = [(0, 0), (1, 1), (2, 2), (0, 1)]
        processes = []

        for i in range(4):
            p = Process(
                target=train_single_model,
                args=(i,
                      self.models[i],
                      torch.device(f'cuda:{i}'),
                      self.dataset,
                      self.crop,
                      self.criterion,
                      selected_channels[i],
                      indices_train,
                      indices_val,
                      filepath,
                      epochs,
                      batch_size,
                      self.verbose)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def train(self, epochs=1001, filepath='', batch_size=128):
        model_i = 0
        np.random.seed(7)
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]
        self.optimizer = [torch.optim.Adam(mdl.parameters(), lr=5e-5) for mdl in self.models]

        # Assign each model to a different GPU
        # self.devices = [torch.device(f'cuda:{i}') for i in range(4)]
        # for i in range(4):
        #     self.models[i].to(self.devices[i])
        #     self.models[i].train()
        self.devices = [torch.device(f'cpu') for i in range(4)]
        self.devices[model_i] = torch.device(f'cuda:{model_i}')
        for i in range(4):
            self.models[i].to(self.devices[i])
            self.models[i].train()

        if self.verbose:
            print('Training...')

        selected_channels = [(0, 0), (1, 1), (2, 2), (0, 1)]
        best_val_loss = [float('inf')] * 4
        # for epoch in trange(epochs):
        #     self.models[0].train()
        #     self.models[1].train()
        #     self.models[2].train()
        #     self.models[3].train()
        #     loss_global = [0, 0, 0, 0]
        #     np.random.shuffle(indices_train)  # Shuffle the indices at the start of each epoch
        #
        #     # Split indices into batches
        #     num_batches = len(indices_train) // batch_size
        #     batches = np.array_split(indices_train, num_batches)
        #
        #     ct = 0
        #     for batch_indices in batches:
        #         ct += 1
        #         # print(ct, '/', len(batches))
        #         # Load batches as chunks
        #         batch = [self.dataset[idx] for idx in batch_indices]
        #         p_t_nu_batch = torch.stack([b[0][:self.crop, :self.crop, :, :] for b in batch])
        #         p_t_nu_vacuum_batch = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in batch])
        #         osc_par_batch = torch.stack([b[3][[0, 1, 2, 5], ] for b in batch])
        #
        #         for mdl in [model_i]:
        #             device = self.devices[mdl]
        #             input_im_i = p_t_nu_vacuum_batch[:, :, :, selected_channels[mdl][0], selected_channels[mdl][1]]
        #             input_im_i = input_im_i[:, None, :, :].to(device)
        #             target_i = p_t_nu_batch[:, :, :, selected_channels[mdl][0], selected_channels[mdl][1]]
        #             target_i = target_i[:, None, :, :].to(device)
        #             osc_par_i = osc_par_batch.to(device)
        #             # Forward pass
        #             output_i = self.models[mdl].forward(x=input_im_i, osc_params=osc_par_i)
        #             loss = self.criterion(output_i, target_i)
        #             loss.backward()
        #             self.optimizer[mdl].step()
        #             self.optimizer[mdl].zero_grad()
        #             loss_global[mdl] += loss.item()
        #
        #     # Validation
        #     self.models[0].eval()
        #     self.models[1].eval()
        #     self.models[2].eval()
        #     self.models[3].eval()
        #
        #     with torch.no_grad():
        #         val_loss_global = [0, 0, 0, 0]
        #         val_num_batches = max(1, len(indices_val) // batch_size)
        #         val_batches = np.array_split(indices_val, val_num_batches)
        #
        #         for batch_indices in val_batches:
        #             batch = [self.dataset[idx] for idx in batch_indices]
        #             p_t_nu_batch = torch.stack([b[0][:self.crop, :self.crop, :, :] for b in batch])
        #             p_t_nu_vacuum_batch = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in batch])
        #             osc_par_batch = torch.stack([b[3][[0, 1, 2, 5]] for b in batch])
        #
        #             for mdl in [model_i]:
        #                 device = self.devices[mdl]
        #                 input_im_i = p_t_nu_vacuum_batch[:, :, :, selected_channels[mdl][0], selected_channels[mdl][1]]
        #                 input_im_i = input_im_i[:, None, :, :].to(device)
        #                 target_i = p_t_nu_batch[:, :, :, selected_channels[mdl][0], selected_channels[mdl][1]]
        #                 target_i = target_i[:, None, :, :].to(device)
        #                 osc_par_i = osc_par_batch.to(device)
        #
        #                 output_i = self.models[mdl].forward(x=input_im_i, osc_params=osc_par_i)
        #                 loss = self.criterion(output_i, target_i)
        #                 val_loss_global[mdl] += loss.item()
        #
        #         val_loss_avg = [vl / len(val_batches) for vl in val_loss_global]
        #
        #         for mdl in [model_i]:
        #             if val_loss_avg[mdl] <= best_val_loss[mdl]:
        #                 best_val_loss[mdl] = val_loss_avg[mdl]
        #                 torch.save(self.models[mdl].state_dict(), f'{filepath}_model{mdl}.pt')
        #                 if self.verbose:
        #                     print(f'>> Saved model {mdl} at epoch {epoch} with val loss {val_loss_avg[mdl]:.4f}')
        #
        #     if epoch % 1 == 0 and self.verbose:
        #         print(f'epoch: {epoch} | '
        #               f'train_loss: {[round(l / len(batches), 4) for l in loss_global]} | '
        #               f'val_loss: {[round(l / len(val_batches), 4) for l in val_loss_global]}')

        batch_size = 1
        for mdl in range(4):
            self.models[mdl].load_state_dict(torch.load(f'{filepath}_model{mdl}.pt', map_location='cuda:0'))

        self.models[0].eval()
        self.models[1].eval()
        self.models[2].eval()
        self.models[3].eval()
        with torch.no_grad():
            val_loss_global = [0, 0, 0, 0]
            val_num_batches = max(1, len(indices_val) // batch_size)
            val_batches = np.array_split(indices_val, val_num_batches)

            for batch_indices in val_batches:
                batch = [self.dataset[idx] for idx in batch_indices]
                p_t_nu_batch = torch.stack([b[0][:self.crop, :self.crop, :, :] for b in batch])
                p_t_nu_vacuum_batch = torch.stack([b[1][:self.crop, :self.crop, :, :] for b in batch])
                osc_par_batch = torch.stack([b[3][[0, 1, 2, 5]] for b in batch])

                osc_pred_batch = p_t_nu_vacuum_batch[0, :, :, :, :].cpu().numpy() * 0
                for mdl in range(4):
                    device = self.devices[mdl]
                    input_im_i = p_t_nu_vacuum_batch[:, :, :, selected_channels[mdl][0], selected_channels[mdl][1]]
                    input_im_i = input_im_i[:, None, :, :].to(device)
                    target_i = p_t_nu_batch[:, :, :, selected_channels[mdl][0], selected_channels[mdl][1]]
                    target_i = target_i[:, None, :, :].to(device)
                    osc_par_i = osc_par_batch.to(device)

                    output_i = self.models[mdl].forward(x=input_im_i, osc_params=osc_par_i)
                    if mdl != 3:
                        osc_pred_batch[:, :, mdl, mdl] = output_i[0, 0, :, :].cpu().numpy()
                    else:
                        osc_pred_batch[:, :, 0, 1] = output_i
                    loss = self.criterion(output_i, target_i)
                    val_loss_global[mdl] += loss.item()

                osc_pred_batch[:, :, 0, 2] = 1 - (osc_pred_batch[:, :, 0, 0] + osc_pred_batch[:, :, 0, 1])
                osc_pred_batch[:, :, 1, 2] = 1 - (osc_pred_batch[:, :, 0, 1] + osc_pred_batch[:, :, 2, 2])
                osc_pred_batch[:, :, 1, 0] = 1 - (osc_pred_batch[:, :, 1, 2] + osc_pred_batch[:, :, 1, 1])
                osc_pred_batch[:, :, 2, 0] = 1 - (osc_pred_batch[:, :, 0, 0] + osc_pred_batch[:, :, 1, 0])
                osc_pred_batch[:, :, 2, 1] = 1 - (osc_pred_batch[:, :, 0, 1] + osc_pred_batch[:, :, 1, 1])

                plot_osc_maps(p_t_nu_batch[0, :, :, :, :], title='Osc. Maps w/ Matter effect')
                plot_osc_maps(osc_pred_batch, title='Pred. Osc. Maps in Vacuum')

            val_loss_avg = [vl / len(val_batches) for vl in val_loss_global]
            print(f'val_loss: {[round(l / len(val_batches), 4) for l in val_loss_avg]}')

    def train_ensemble(self, ensemble_size=1, epochs=1001):
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
            self.train(epochs=epochs, filepath=f)


if __name__ == '__main__':
    complexities = [3]
    for c in complexities:
        print("Training ensemble of models of complexity ", c)
        modell = TrainModel(complexity=c, plotR=True, verbose=True)
        modell.train_ensemble(ensemble_size=1, epochs=1000)
