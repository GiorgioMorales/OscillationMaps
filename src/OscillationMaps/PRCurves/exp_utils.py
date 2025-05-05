"""
Utlity functions for experiments and their visualisation
Get folder names for storing experiments results

The experiments results are located in /root-folder/feature_dim{feature_dim}/loc_real{loc_real}/scale_real{scale_real}/loc_fake{loc_fake}/scale_fake{scale_fake}/
                                            split{train_test_ratio}/outliers_real{outliers_real}/outliers_fake{outliers_fake}
"""
import re
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import shutil
from glob import glob

from OscillationMaps.PRCurves.distributions import create_distrib
from OscillationMaps.PRCurves.iipr import (
    get_ground_truth_classifiers,
    get_prd,
    DistanceMatrix,
    get_ipr_classifiers,
    get_knn_classifiers,
    get_ppr_classifiers,
    get_parzen_classifiers,
    get_coverage_classifiers
)

from OscillationMaps.PRCurves.utils import get_slopes

ALL_METHODS = ["ipr", "knn", "parzen", "coverage"]
ALL_METHODS_AND_GT = ALL_METHODS + ["ground-truth"]
NUM_SAMPLES_GT = 100_000


def convert_samples_to_indexes(real_samples, fake_samples):
    """
    Using the DistanceMatrix class from src.iipr, we access values by index
    This function takes the samples and outputs what will be necessary for both DistanceMatrix and the
    get_prd function
    """
    all_samples = np.concatenate([real_samples, fake_samples], axis=0)
    num_real = real_samples.shape[0]
    num_fake = fake_samples.shape[0]
    all_real_indices = np.arange(num_real)
    all_fake_indices = np.arange(num_fake) + num_real
    return all_samples, all_real_indices, all_fake_indices


def get_data_params_folder(distribution_type, real_params, fake_params, feature_dim):
    if distribution_type == "gaussian":
        loc_real, scale_real = real_params
        loc_fake, scale_fake = fake_params
        folder_name = f"feature_dim{feature_dim}/loc_real{loc_real}/scale_real{scale_real}/loc_fake{loc_fake}/scale_fake{scale_fake}"
        return folder_name


def get_params_from_folder(folder_name):
    feature_dim = re.findall(r"feature_dim(\d+\.*\d*)", folder_name)[0]
    loc_real = re.findall(r"loc_real(\d+\.*\d*)", folder_name)[0]
    scale_real = re.findall(r"scale_real(\d+\.*\d*)", folder_name)[0]
    loc_fake = re.findall(r"loc_fake(\d+\.*\d*)", folder_name)[0]
    scale_fake = re.findall(r"scale_fake(\d+\.*\d*)", folder_name)[0]
    train_test_ratio = re.findall(r"train_test_ratio(\d+\.*\d*)", folder_name)[0]
    outliers_real = re.findall(r"outliers_real(\d+\.*\d*)", folder_name)[0]
    outliers_fake = re.findall(r"outliers_fake(\d+\.*\d*)", folder_name)[0]
    params = dict(
        feature_dim=feature_dim,
        loc_real=loc_real,
        scale_real=scale_real,
        loc_fake=loc_fake,
        scale_fake=scale_fake,
        train_test_ratio=train_test_ratio,
        outliers_real=outliers_real,
        outliers_fake=outliers_fake,
    )
    # convert the params by removing misleading endings .0 : 2.0 --> 2
    # params = {key: f"{value:g}" for key,value in params.items()}
    return params


def get_exp_folder_name(
        distribution_type,
        real_params,
        fake_params,
        train_test_ratio,
        outliers_real,
        outliers_fake,
        feature_dim,
):
    """Get the corresponding folder to some experiment parameters"""
    if distribution_type == "gaussian":
        distr_params_folder = get_data_params_folder(
            distribution_type=distribution_type,
            real_params=real_params,
            fake_params=fake_params,
            feature_dim=feature_dim,
        )
        folder_name = (
                distr_params_folder
                + f"/train_test_ratio{train_test_ratio}/outliers_real{outliers_real}/outliers_fake{outliers_fake}"
        )
        return folder_name


def get_indices_and_dist_matrix(real_distrib,
                                fake_distrib,
                                params_P,
                                params_Q,
                                feature_dim,
                                num_samples,
                                outliers_real,
                                outliers_fake,
                                device,
                                approx):
    """Sample points, add outliers, and create distance matrix
    From samples compute distance matrix and return the indices of real and fake data
    """
    real_distrib_object = create_distrib(real_distrib, params=params_P, feature_dim=feature_dim)
    fake_distrib_object = create_distrib(fake_distrib, params=params_Q, feature_dim=feature_dim)

    real_samples = real_distrib_object.sample(num_samples)
    fake_samples = fake_distrib_object.sample(num_samples)

    real_samples = introduce_outliers(outliers_real, real_samples)
    fake_samples = introduce_outliers(outliers_fake, fake_samples)

    all_samples = np.concatenate([real_samples, fake_samples], axis=0)
    num_real = real_samples.shape[0]
    num_fake = fake_samples.shape[0]
    all_real_indices = np.arange(num_real)
    all_fake_indices = np.arange(num_fake) + num_real

    distance_matrix = DistanceMatrix(samples=all_samples,
                                     device=device,
                                     approx=approx,
                                     real_distrib_object=real_distrib_object,
                                     fake_distrib_object=fake_distrib_object)

    return all_real_indices, all_fake_indices, distance_matrix


def split_all_data(real_samples, fake_samples, split_train_test_ratio):
    """split data if split ratio is coherent else return train and test samples = the input samples"""
    if split_train_test_ratio and (0 < split_train_test_ratio < 1):
        # Dealing in the else statement with the extreme cases where split train test ==0 or 1 which have no sense in the experiments
        real_samples_train, real_samples_test = train_test_split(
            real_samples, train_size=split_train_test_ratio
        )
        fake_samples_train, fake_samples_test = train_test_split(
            fake_samples, train_size=split_train_test_ratio
        )
        # The value of split_train_test_ratio should be between 0 and 1
        # Value equal to 0 would mean not fitting the metrics and evaluating the unfitted metrics on the entire dataset
        # Conversely value equal to 1 would mean that
        # By default, if value = 0 --> no train test split
    else:
        real_samples_train = real_samples_test = real_samples
        fake_samples_train = fake_samples_test = fake_samples
    return (
        real_samples_train,
        real_samples_test,
        fake_samples_train,
        fake_samples_test,
    )


def introduce_outliers(outliers, samples_train):
    """Adds outliers in the samples. We choose to set the values of outliers to 4.
    This means that the norm of outliers is \sqrt{dimension}.4"""
    if outliers:
        samples_train[:outliers, :] = 4
    return samples_train


def get_classifiers(method, real_samples_train, fake_samples_train, nearest_k, gammas, distance_matrix):
    if method not in ALL_METHODS:
        raise Exception(
            f"Method should be any of {ALL_METHODS} but got {method} instead"
        )
    elif method == "ipr":
        classifiers = get_ipr_classifiers(
            real_samples_train=real_samples_train,
            fake_samples_train=fake_samples_train,
            nearest_k=nearest_k,
            gammas=gammas,
            distance_matrix=distance_matrix
        )
    elif method == "knn":
        classifiers = get_knn_classifiers(
            real_samples_train=real_samples_train,
            fake_samples_train=fake_samples_train,
            nearest_k=nearest_k,
            gammas=gammas,
            distance_matrix=distance_matrix
        )

    elif method == "ppr":
        classifiers = get_ppr_classifiers(
            real_samples_train=real_samples_train,
            fake_samples_train=fake_samples_train,
            nearest_k=nearest_k,
            gammas=gammas,
            distance_matrix=distance_matrix
        )

    elif method == "parzen":
        classifiers = get_parzen_classifiers(
            real_samples_train=real_samples_train,
            fake_samples_train=fake_samples_train,
            nearest_k=nearest_k,
            gammas=gammas,
            distance_matrix=distance_matrix
        )

    elif method == "coverage":
        classifiers = get_coverage_classifiers(
            real_samples_train=real_samples_train,
            fake_samples_train=fake_samples_train,
            nearest_k=nearest_k,
            gammas=gammas,
            distance_matrix=distance_matrix
        )

    return classifiers


def save_npy_and_params(method, output_folder, arguments_dict, alphas_betas):
    np.save(os.path.join(output_folder, "alphas_betas_" + method), alphas_betas)
    with open(os.path.join(output_folder, "arguments.json"), "w") as file:
        json.dump(arguments_dict, file)


def remove_files_folder(folder_path):
    """
    Remove all files and sub folders from a path in order to run experiment
    """
    if os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        print(f"Removed previous experiments from {folder_path}")


def get_prd_curve_experiment(
        number_angles,
        nearest_k,
        method,
        split_train_test_ratio,
        distance_matrix,
        all_real_indices,
        all_fake_indices,
        random_seed=None
):
    """
    Run experiment between 2 distributions, returns the PRD curves in a numpy array and dict of arguments
    The distributions are "present" inside the distance_matrix object which stores the distances between
    all the points.
    """
    arguments_dict = {"number_angles": number_angles,
                      "nearest_k": nearest_k,
                      "method": method,
                      "split_train_test_ratio": split_train_test_ratio,
                      "random_seed": random_seed}

    slopes = get_slopes(num_angles=number_angles)

    ###############################################################################
    # define the classifier class on which the optimisation process will be held

    if method == "ground-truth":
        # "overwrite the nb of samples and the samples themselves that are used for the other methods
        # as in for ground truth we compute on a larger number of points"
        real_samples = distance_matrix.real_distrib_object.sample(NUM_SAMPLES_GT)
        fake_samples = distance_matrix.fake_distrib_object.sample(NUM_SAMPLES_GT)

        all_samples = np.concatenate([real_samples, fake_samples], axis=0)
        num_real = real_samples.shape[0]
        num_fake = fake_samples.shape[0]
        idx_real_samples_test = np.arange(num_real)
        idx_fake_samples_test = np.arange(num_fake) + num_real

        classifiers = get_ground_truth_classifiers(real_distrib=distance_matrix.real_distrib_object,
                                                   fake_distrib=distance_matrix.fake_distrib_object,
                                                   samples=all_samples)
    elif method in ALL_METHODS:

        # Adding outliers if necessary
        # Adding outliers of norm 10

        idx_real_samples_train, idx_real_samples_test, idx_fake_samples_train, idx_fake_samples_test = \
            split_all_data(all_real_indices, all_fake_indices, split_train_test_ratio)
        classifiers = get_classifiers(method=method,
                                      real_samples_train=idx_real_samples_train,
                                      fake_samples_train=idx_fake_samples_train,
                                      nearest_k=nearest_k,
                                      gammas=slopes,
                                      distance_matrix=distance_matrix)
    else:
        raise Exception(
            f"Method should be any of {ALL_METHODS_AND_GT} but got {method} instead"
        )

    ####################################################################
    # Run the experiment on sampled elements representing test data

    # optimisation problem on the relevent classifiers
    alphas_betas = get_prd(
        lambdas=slopes,
        real_samples=idx_real_samples_test,
        fake_samples=idx_fake_samples_test,
        classifiers=classifiers,
    )

    return alphas_betas, arguments_dict


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    e.g.
    dot_dictionnary = dotdict({"a":5,"b":np.linspace(1,10,12)})
    print(dot_dictionnary.a)
    dot_dictionnary.c = "new string"
    
    """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def check_args(args):
    """
    Sanity check on args parsed with arg_parse
    """
    if len(args.methods) != len(args.split_ratios):
        raise Exception(
            f"There should be as many methods as split ratios but got resp {len(args.methods)} and {len(args.split_ratios)}")
    if "ground-truth" in args.methods:
        ratios_gt = [ratio for ratio, method in zip(args.split_ratios, args.methods) if method == "ground-truth"]
        if 0 in ratios_gt:
            print("There should not be any split ratios for ground-truth. Will be ignoring them.")
        if args.compute_GT:
            print(
                "Ground truth was asked to be computed twice, with arguments compute_GT and in the methods.\nWill be skipping ground-truth in the methods")
            filtered_lists = [(method, split_ratio) for method, split_ratio in zip(args.methods, args.split_ratios) if
                              method != "ground-truth"]
            args.methods, args.split_ratios = zip(*filtered_lists)


def extract_metric_vgg_exp(method, metric, filenames):
    # List to store metric values from each JSON file
    metric_values = []

    # Iterate over each JSON file
    for filename in filenames:
        # Read JSON file
        with open(filename, 'r') as f:
            data = json.load(f)

        # Extract the metric value from the JSON data
        metric_value = data[method][metric]

        # Append the metric value to the list
        metric_values.append(metric_value)

    return metric_values


def get_metrics_dict_vgg(exp_path):
    """
    take all the metrics associated with exp path and combine into one dict
    """
    filenames = sorted(glob(os.path.join(exp_path, "*/*.json")))
    truncs = [float(os.path.basename(os.path.dirname(filename))[-3:]) for filename in filenames]

    dic_metrics = {}

    methods = ["ipr", "knn", "parzen", "coverage"]
    metrics = [
        'max_precision',
        'max_recall',
        'median_precision',
        'median_recall',
        'fbeta_precision',
        'fbeta_recall',
        'auc'
    ]

    for method in methods:
        dic_metrics[method] = {}
        for metric in metrics:
            dic_metrics[method][metric] = extract_metric_vgg_exp(method, metric, filenames)
    return dic_metrics, filenames, truncs


def main_test():
    feature_dim = 1
    loc_real = 203
    loc_fake = 3.5
    scale_fake = 4
    scale_real = 5
    train_test_ratio = 6
    outliers_real = 7
    outliers_fake = 8

    test_folder = (
        f"/feature_dim{feature_dim}/loc_real{loc_real}/scale_real{scale_real}/loc_fake{loc_fake}/scale_fake{scale_fake}"
        f"/train_test_ratio{train_test_ratio}/outliers_real{outliers_real}/outliers_fake{outliers_fake}"
    )
    print(f"test_folder: {test_folder}")
    params = get_params_from_folder(test_folder)
    print(f"The params found for folder {test_folder} are:\n{params}")


if __name__ == "__main__":
    main_test()
