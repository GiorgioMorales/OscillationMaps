import numpy as np
from OscillationMaps.Data.DataLoader import *
# from PredictionIntervals.Trainer.TrainNN import Trainer
import matplotlib.pyplot as plt

from EquationLearning.SymbolicRegressor.MSSP import *
from EquationLearning.SymbolicRegressor.SetGAP import SetGAP
from EquationLearning.Trainer.TrainNNmodel import Trainer

from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    print(torch.cuda.is_available())

    H5_file_path = "oscillation_maps_extended"
    dataset = HDF5Dataset(H5_file_path)

    indices = np.arange(len(dataset))
    batch_size = 16
    num_batches = len(indices) // batch_size
    batches = np.array_split(indices, num_batches)

    all_sum_channel = []
    all_osc_par = []
    for i in range(len(batches)):
        batch = [dataset[idx] for idx in batches[i]]
        p_t_nu_batch = np.stack([b[0] for b in batch])
        p_t_nu_batch = p_t_nu_batch[:, :, :, 0, 1]
        sum_channel = np.max(p_t_nu_batch, axis=(1, 2)) - np.min(p_t_nu_batch, axis=(1, 2))  # np.mean(p_t_nu_batch, axis=(1, 2))  #
        osc_par_batch = np.stack([b[3] for b in batch])

        all_sum_channel += list(sum_channel)
        all_osc_par += list(osc_par_batch)

    all_sum_channel = np.array(all_sum_channel)
    all_osc_par = np.array(all_osc_par)

    indices = np.arange(len(all_sum_channel))
    np.random.seed(7)
    np.random.shuffle(indices)
    train_ind, val_ind = indices[:int(.9 * len(indices))], indices[int(.9 * len(indices)):]

    all_osc_par = all_osc_par[:, np.array([0, 1, 2, 5])]  # np.array([0, 1, 2])
    X = all_osc_par[train_ind, :]
    Y = all_sum_channel[train_ind]
    Xval = all_osc_par[val_ind, :]
    Yval = all_sum_channel[val_ind]

    names = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5']
    types = ['continuous'] * 6
    x_scaler = StandardScaler()
    dataset = InputData(X=x_scaler.fit_transform(X), Y=Y, names=names, types=types)

    predictor = Trainer(dataset=dataset, modelType='NN3')
    predictor.train(batch_size=64, epochs=50000, printProcess=True)
    Yval_pred = predictor.model.evaluateFold(valxn=x_scaler.transform(Xval))
    plt.figure()
    plt.scatter(Yval, Yval_pred)
    plt.show()

    regressor = SetGAP(dataset=dataset, bb_model=predictor.model, n_candidates=3)
    results = regressor.run()


    # trainer = Trainer(X=all_osc_par[train_ind, :], Y=all_sum_channel[train_ind],
    #                   Xval=all_osc_par[val_ind, :], Yval=all_sum_channel[val_ind], method='MCDropout')
    # trainer.train(printProcess=True)
    # # torch.save(trainer.model.model.network.state_dict(), 'src/OscillationMaps/Models/saved_models/NN_params.pth')
    # Yval_pred, _, _ = trainer.evaluate(all_osc_par[val_ind, :], normData=True)
    # mse = np.mean((all_sum_channel[val_ind] - Yval_pred) ** 2)
    # print("Mean Squared Error:", mse)
    # plt.figure()
    # plt.scatter(all_sum_channel[val_ind], Yval_pred)
    # plt.show()


    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(all_osc_par[:, 0], all_sum_channel)
    # plt.show()

    # import shap
    #
    # Xeval, Yeval = trainer._apply_normalization(all_osc_par[val_ind, :], all_sum_channel[val_ind])
    # Xval_tensor = torch.tensor(Xeval, dtype=torch.float32).cuda()
    # trainer.model.model.network.eval()
    #
    # # SHAP requires a wrapper to work with PyTorch models
    # explainer = shap.DeepExplainer(trainer.model.model.network, Xval_tensor[:1000])  # Use a small sample for background
    # shap_values = explainer.shap_values(Xval_tensor[1000:1900, :])  # Evaluate on a few samples
    # shap.plots.violin(shap_values, features=Xeval, plot_type="layered_violin")
    # shap.summary_plot(shap_values[:, :, 0], Xeval[1000:1900, :])
