import os
from tqdm import trange
import torch.optim as optim
from OscillationMaps.utils import *
from OscillationMaps.Models.model import *
# from OscillationMaps.Models.unet import *
from OscillationMaps.Data.DataLoader import HDF5Dataset


class TrainModel:
    def __init__(self, datapath='oscillation_maps_extended', modelType='DDPM', complexity=1, verbose=False, plotR=False):
        self.modelType = modelType
        self.complexity = complexity
        self.dataset = HDF5Dataset(datapath)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.reset_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=2e-3)
        self.criterion = nn.MSELoss()
        self.verbose = verbose
        self.plot = plotR
        self.T = 100

    def reset_model(self):
        return DenoisingNet(complexity=self.complexity).to(self.device)
        # return UNet(
        #     image_size=60,
        #     in_channel=9,
        #     inner_channel=32,
        #     out_channel=9,
        #     res_blocks=2,
        #     attn_res=[4]
        # ).to(self.device)

    def train(self, epochs=1001, filepath='', batch_size = 16):
        indices = np.arange(len(self.dataset))
        np.random.shuffle(indices)
        indices_train = indices[:int(len(indices) * 0.8)]
        indices_val = indices[int(len(indices) * 0.8):]

        for epoch in trange(epochs):
            loss_global = 0
            np.random.shuffle(indices_train)  # Shuffle the indices at the start of each epoch

            # Split indices into batches
            num_batches = len(indices_train) // batch_size
            batches = np.array_split(indices_train, num_batches)

            for batch_indices in batches:
                # Load batches as chunks
                batch = [self.dataset[idx] for idx in batch_indices]
                p_t_nu_batch = torch.stack([b[0][:60, :60, :, :] * 255 for b in batch]).to(self.device)  # (batch_size, 120, 120, 3, 3)
                p_t_nu_vacuum_batch = torch.stack([b[1][:60, :60, :, :] * 255 for b in batch]).to(self.device)  # (batch_size, 120, 120, 3, 3)

                # Generate noise/noisy samples
                noise = torch.randn_like(p_t_nu_batch).to(self.device)
                t = torch.randint(0, self.T, (p_t_nu_batch.size(0),), device=self.device, dtype=torch.float32).view(-1, 1)
                noisy_input = forward_diffusion(p_t_nu_batch, t.long().squeeze().to(self.device), noise, timesteps=self.T)

                # Forward pass
                # self.model = torch.load(filepath)  # TODO
                noisy_input = noisy_input.permute(0, 3, 4, 1, 2)
                p_t_nu_vacuum_batch = p_t_nu_vacuum_batch.permute(0, 3, 4, 1, 2)
                outputs = self.model(x=noisy_input, vacuum=p_t_nu_vacuum_batch, t=t)

                # Backpropagation
                loss = self.criterion(outputs, p_t_nu_batch.permute(0, 3, 4, 1, 2))
                # b, h, w, c, d = noisy_input.shape
                # noisy_input = noisy_input.permute(0, 3, 4, 1, 2).view(b, c * d, h, w)
                # # p_t_nu_vacuum_batch = p_t_nu_vacuum_batch.permute(0, 3, 4, 1, 2).view(b, c * d, h, w)
                # # outputs = self.model(x=noisy_input, vacuum=p_t_nu_vacuum_batch, t=t)
                # outputs = self.model(noisy_input, t)
                #
                # # Backpropagation
                # loss = self.criterion(outputs, p_t_nu_batch.permute(0, 3, 4, 1, 2).view(b, c * d, h, w))
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_global += loss.item()

            torch.save(self.model, filepath)

            if epoch % 1 == 0 and self.verbose:
                print('epoch:', epoch, 'loss:', loss_global / len(batches))

        # Analyze results
        self.model = torch.load(filepath)
        osc_maps_gen = []
        indices_val = indices_val[0:2]
        val_data = [self.dataset[idx] for idx in indices_val]
        # p_t_nu_val = torch.stack([b[0][:60, :60, :, :] for b in val_data]).to(self.device)
        p_t_nu_vacuum_val = torch.stack([b[1][:60, :60, :, :] for b in val_data]).to(self.device)
        num_samples = p_t_nu_vacuum_val.shape[0]
        batch_size_val, total_size = 10, 0
        while total_size < num_samples:
            num_gen = num_samples - len(osc_maps_gen)
            for start_idx in range(0, num_gen, batch_size):
                end_idx = min(start_idx + batch_size, num_gen)
                x_seq = denoise(self.model, timesteps=self.T,
                                noisy_input=torch.randn_like(p_t_nu_vacuum_val[start_idx:end_idx, :, :, :, :].permute(0, 3, 4, 1, 2).to(self.device)),
                                vacuum=p_t_nu_vacuum_val[start_idx:end_idx, :, :, :, :].permute(0, 3, 4, 1, 2).to(self.device))
                osc_maps_gen_batch = x_seq.cpu().detach().permute(0, 3, 4, 1, 2).numpy()
                osc_maps_gen_batch = osc_maps_gen_batch[np.unique(np.where(~np.isnan(osc_maps_gen_batch))[0]), :]
                osc_maps_gen.append(osc_maps_gen_batch)
            total_size = len(np.vstack(osc_maps_gen))

        osc_maps_gen = np.vstack(osc_maps_gen)
        if self.plot:
            plot_osc_maps(osc_maps_gen[0, :, :, :, :], title='Osc. Maps w/ Matter effect')
            plot_osc_maps(p_t_nu_vacuum_val.cpu()[0, :, :, :, :], title='Osc. Maps in Vacuum')

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
        model.train_ensemble(ensemble_size=1, epochs=50)
