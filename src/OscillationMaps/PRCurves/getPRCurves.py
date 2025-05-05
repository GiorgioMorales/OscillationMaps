import os
import json
import torch
import argparse
import numpy as np

from OscillationMaps.PRCurves.utils import get_slopes
from OscillationMaps.PRCurves.iipr import get_prd, DistanceMatrix
from OscillationMaps.PRCurves.exp_utils import get_classifiers, split_all_data
from OscillationMaps.PRCurves.scores import get_extreme_values, prd_to_median_pair, prd_to_max_f_beta_pair, prd_to_auc


"""
Create Precision and Recall curves based on .npy files of embeddings

python exp_real.py --real-data /home/sykes232/Code/PrecisionRecall/iipr-stylegan-exp/stylegan-embeddings/ffhq_dataset/image_embedding_inception_model.pt\
    --fake-data /home/sykes232/Code/PrecisionRecall/iipr-stylegan-exp/stylegan-embeddings/0.1/image_embedding_inception_model.pt\
        --num-samples 1000\
            --split-train-test-ratio 0.5\
                --number-angles 11\
                    --nearest-k 3\
                        --output-folder /tmp/test-real-exp\
                            --methods ipr\
                                --device cuda
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-data", type=str, required=True,
                        help="The type of distribution which will be used for real data. Can be any of ['gaussian','uniform-sphere']", )
    parser.add_argument("--fake-data", type=str, required=True,
                        help="The type of distribution which will be used for real data. Can be any of ['gaussian','uniform-sphere']", )
    parser.add_argument("--num-samples", required=True, type=int)

    parser.add_argument("--split-train-test-ratio", required=True, type=float)

    parser.add_argument("--methods", required=True, type=str, nargs="+")
    parser.add_argument("--number-angles", required=True, type=int)
    parser.add_argument("--nearest-k", required=True, type=int)
    parser.add_argument("--c-dist-approx", action="store_true")
    parser.add_argument("--output-folder", type=str, help="Folder where to save the experiments numpy files",
                        required=True)
    parser.add_argument("--device", required=True, type=str)

    return parser.parse_args()


def get_prd_real_data(distance_matrix,
                      idx_real_samples_train,
                      idx_fake_samples_train,
                      idx_real_samples_test,
                      idx_fake_samples_test,
                      number_angles,
                      nearest_k,
                      method,
                      ):
    """Run experiment between 2 distributions, returns the PRD curves in a numpy array"""

    arguments_dict = locals()

    slopes = get_slopes(num_angles=number_angles)

    ###############################################################################
    # define the classifier class on which the optimisation process will be held
    if method in ["ipr", "knn", "ppr", "parzen", "coverage"]:

        classifiers = get_classifiers(method=method,
                                      real_samples_train=idx_real_samples_train,
                                      fake_samples_train=idx_fake_samples_train,
                                      nearest_k=nearest_k,
                                      gammas=slopes,
                                      distance_matrix=distance_matrix)
    else:
        raise Exception(f"Method should be any of ['ground-truth','ipr', 'knn','ppr'] but got {method} instead")

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


def open_embedding(embedding_file_path):
    if embedding_file_path.endswith(".pt"):
        embeddings = torch.load(embedding_file_path)
        return embeddings.detach().cpu().numpy().squeeze()
    elif embedding_file_path.endswith(".npy"):
        return np.load(embedding_file_path).squeeze()
    else:
        raise Exception(f"Embeddings should be either numpy arrays or torch tensors but the"
                        f"input path is {embedding_file_path}")


def getPRCurves(methods,
                real_data,
                fake_data,
                num_samples,
                split_train_test_ratio,
                number_angles,
                nearest_k,
                c_dist_approx,
                output_folder,
                device,
                output_file_desc=None):
    argss = {
        "methods": methods,
        "real_data": real_data,
        "fake_data": fake_data,
        "num_samples": num_samples,
        "split_train_test_ratio": split_train_test_ratio,
        "number_angles": number_angles,
        "nearest_k": nearest_k,
        "c_dist_approx": c_dist_approx,
        "output_folder": output_folder
    }
    metrics = {}

    dict_PRD = {}

    if not (isinstance(real_data, torch.Tensor) or isinstance(real_data, np.ndarray)):
        real_data = open_embedding(real_data)
    if not (isinstance(fake_data, torch.Tensor) or isinstance(fake_data, np.ndarray)):
        fake_data = open_embedding(fake_data)

    os.makedirs(output_folder, exist_ok=True)

    metrics["arguments"] = {
        'num_samples': argss["num_samples"],
        'split_train_test_ratio': argss["split_train_test_ratio"],
        'number_angles': argss["number_angles"],
        'nearest_k': argss["nearest_k"],
        'c_dist_approx': argss["c_dist_approx"],
        'output_folder': argss["output_folder"]
    }

    real_samples = real_data[-num_samples:]
    fake_samples = fake_data[:num_samples]

    print(f"{real_data.shape=} & {fake_data.shape=}")

    print(f"{real_samples.shape=} & {fake_samples.shape=}")
    all_samples = np.concatenate([real_samples, fake_samples], axis=0)
    num_real = num_samples
    num_fake = num_samples

    all_real_indices = np.arange(num_real)
    all_fake_indices = np.arange(num_fake) + num_real

    distance_matrix = DistanceMatrix(samples=all_samples,
                                     approx=c_dist_approx,
                                     real_distrib_object=None,
                                     fake_distrib_object=None,
                                     device=device)

    idx_real_samples_train, idx_real_samples_test, idx_fake_samples_train, idx_fake_samples_test = \
        split_all_data(all_real_indices, all_fake_indices, split_train_test_ratio)
    # idx_real_samples_train, idx_real_samples_test, idx_fake_samples_train, idx_fake_samples_test = \
    #     all_real_indices[0:int(len(num_real) / 2)], all_fake_indices[0:int(len(num_real) / 2)], \
    #     all_real_indices[int(len(num_real) / 2):], all_fake_indices[int(len(num_real) / 2):],
    for method in methods:
        print(f"##################### Running on method {method} #####################\n\n")
        argss["method"] = method

        alphas_betas, arguments_dict = get_prd_real_data(distance_matrix=distance_matrix,
                                                         idx_fake_samples_test=idx_fake_samples_test,
                                                         idx_fake_samples_train=idx_fake_samples_train,
                                                         idx_real_samples_test=idx_real_samples_test,
                                                         idx_real_samples_train=idx_real_samples_train,
                                                         number_angles=number_angles,
                                                         nearest_k=nearest_k,
                                                         method=method)
        dict_PRD[method] = alphas_betas
        ##########################################################
        # COMPUTE SCALAR METRICS
        precision, recall = alphas_betas[:, 0], alphas_betas[:, 1]

        max_precision, max_recall = get_extreme_values(precision=precision, recall=recall)
        median_precision, median_recall = prd_to_median_pair(precision=precision, recall=recall)
        fbeta_recall, fbeta_precision = prd_to_max_f_beta_pair(precision=precision, recall=recall, beta=8)
        auc = prd_to_auc(precision, recall)

        metrics[method] = {
            "max_precision": max_precision,
            "max_recall": max_recall,
            "median_precision": median_precision,
            "median_recall": median_recall,
            "fbeta_precision": fbeta_precision,
            "fbeta_recall": fbeta_recall,
            "auc": auc
        }

        if not output_file_desc:
            output_file_desc = ""

        array_path = os.path.join(output_folder, f"alphas_betas_{output_file_desc}_m_{method}.npy")

        np.save(array_path, alphas_betas)
        print(f'Saved prd in {array_path}')

    with open(os.path.join(output_folder, f"metrics_{output_file_desc}.json"), "w") as f:
        json.dump(metrics, f)
    return dict_PRD


# if __name__ == "__main__":
#     args = parse_args()
#     main(
#         real_data=args.real_data,
#         fake_data=args.fake_data,
#         num_samples=args.num_samples,
#         split_train_test_ratio=args.split_train_test_ratio,
#         number_angles=args.number_angles,
#         nearest_k=args.nearest_k,
#         c_dist_approx=args.c_dist_approx,
#         output_folder=args.output_folder,
#         methods=args.methods,
#         device=args.device
#     )
