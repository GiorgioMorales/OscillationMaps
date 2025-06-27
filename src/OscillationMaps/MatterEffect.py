import os
from OscillationMaps.Models.MLPs import *
from OscillationMaps.utils import get_project_root
from OscillationMaps.VacuumMaps import get_oscillation_maps_vacuum


class MatterEffect:
    def __init__(self):
        """Class used to produce 9 oscillation maps estimating matter effect"""
        self.crop = 120
        self.model_is = np.arange(9)
        self.n_models = len(self.model_is)
        self.simpler_models = [0, 6, 8]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_mean, self.dataset_std = [0]*self.n_models, [0]*self.n_models
        self.models = self.reset_model()
        self.channels = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]

        # Create positional grid
        x_coords = torch.arange(self.crop).view(-1, 1).repeat(1, self.crop).flatten()
        y_coords = torch.arange(self.crop).repeat(self.crop)
        pos_grid = torch.stack([x_coords, y_coords], dim=1)
        pos_grid = pos_grid.repeat(1, 1).repeat_interleave(1, dim=0).repeat(1, 1)
        self.pos_grid = pos_grid.expand(self.crop * self.crop, 2)

        # Load models
        root = get_project_root()
        folder = os.path.join(root, "Models//saved_models//MLPS")
        for im, mdl in enumerate(self.model_is):
            f = folder + "//Model--Instance0.pth"
            self.models[im].to(self.device)
            stats = torch.load(f'{f}_NNmodel{mdl}_norm_stats.pt', map_location=self.device)
            self.dataset_mean[im] = stats['mean']
            self.dataset_std[im] = stats['std']
            self.models[im].load_state_dict(torch.load(f'{f}_NNmodel{mdl}.pt', map_location=self.device))
            self.models[im].eval()

    def reset_model(self):
        models = []
        for i in self.model_is:
            sin_flag, num_feat = False, 9
            if i == 1:
                models.append(MLP3(input_features=num_feat))
            else:
                if i in self.simpler_models:
                    sin_flag, num_feat = True, 8
                models.append(MLP4(input_features=num_feat, sin=sin_flag))
        return models

    def get_maps(self, osc_pars: np.ndarray):
        """
        :param osc_pars: Batches of oscillation parameters in order: [theta12, theta23, theta13, delta_cp, m21, m31]
        :return: 9 oscillation maps as a 3x3 np.array
        """
        if osc_pars.ndim == 1:
            osc_pars = osc_pars[None, :]

        osc_pars = torch.from_numpy(osc_pars)
        osc_pred_batch = np.zeros((osc_pars.shape[0], self.crop, self.crop, 3, 3))
        with torch.no_grad():
            for n in range(osc_pars.shape[0]):
                # Repeat osc params
                osc_rep = osc_pars[n, :].repeat(self.crop * self.crop, 1)
                # Get maps in vacuum
                vacuum_maps = get_oscillation_maps_vacuum(osc_pars=osc_pars[n, :])

                for im, mdl in enumerate(self.model_is):
                    ch_i, ch_j = self.channels[mdl]
                    if mdl in self.simpler_models:
                        input_vector = torch.cat([self.pos_grid, osc_rep], dim=1).to(self.device)
                    else:
                        input_vacuum = torch.from_numpy(vacuum_maps[:, :, ch_i, ch_j].reshape(-1, 1))
                        input_vector = torch.cat([self.pos_grid, osc_rep, input_vacuum], dim=1).to(self.device)
                    input_vector = (input_vector - self.dataset_mean[im]) / self.dataset_std[im]
                    output_i = torch.clip(self.models[im](input_vector.float().to(self.device)), 0, 1)
                    osc_pred_batch[n, :, :, ch_i, ch_j] = np.reshape(output_i.cpu().numpy(), (self.crop, self.crop))

        return osc_pred_batch


if __name__ == '__main__':
    # Define parameters
    osc_pars_in = np.array([2.392e+02,  2.955e+02,  2.183e+02,  2.128e+02,  6.523e-05, -5.502e-03])

    # Propagate
    propagator = MatterEffect()
    maps_out = propagator.get_maps(osc_pars=osc_pars_in)

    # Plot
    from OscillationMaps.utils import plot_osc_maps
    for ii in range(maps_out.shape[0]):
        plot_osc_maps(maps_out[ii], title='Predicted Oscillation Maps: ' + str(ii + 1))
