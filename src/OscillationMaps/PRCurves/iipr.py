"""
Unifying approach for Precision and Recall metrics. Heart of the code where the sets of binary classifiers are defined 
The dual approach optimisation is made on those binary classifiers 
"""
import gc
from typing import Any
import numpy as np
import os
import sys
from math import ceil

import torch

from OscillationMaps.PRCurves.prdc import get_kth_value
from OscillationMaps.PRCurves.utils import compute_fpr_fnr

# importing the modules from src
SCRIPT_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_SRC_DIR + "/..")


# batch size used in the batched version of the distance matrix computation
BATCH_SIZE_DIST_MATRIX = 1000


class DistanceMatrix(object):
    def __init__(self, samples, device, approx, real_distrib_object=None, fake_distrib_object=None):
        """
        Allows to run an experiment on several methods without having to compute the distance matrixes 
        repeatedly. If we want to compute the ground-truth curve, we must pass as an argument the real
        and fake distribution objects to the DistanceMatrix __init__() 
        approx: compute_mode in cdist, can use matrix multiplication to accelerate computation
        """
        self.samples = samples
        self.device = device
        self.approx = approx

        # these are used only when computing ground truth
        self.real_distrib_object = real_distrib_object
        self.fake_distrib_object = fake_distrib_object

        self.distances = compute_pairwise_distance_batched(data_x=samples, device=device, approx=approx)
        # with matrix multiplication approximation distances from points to themselves are not always 0
        np.fill_diagonal(self.distances, 0)

    def __call__(self, indices_x, indices_y):
        if not isinstance(indices_x, np.ndarray):
            indices_x = np.array(indices_x)
        if not isinstance(indices_y, np.ndarray):
            indices_y = np.array(indices_y)
        return self.distances[indices_x[:, np.newaxis], indices_y[np.newaxis, :]]

    def get_knn_radius(self, indices_x, indices_y, nearest_k):
        """In the case without split, we should be setting nearest_k to the value we expect + 1 in order
        to exclude the point to itself, having a distance of 0"""
        distances_x_to_y = self.__call__(indices_x, indices_y)
        radii = get_kth_value(distances_x_to_y, k=nearest_k, axis=-1)
        return radii

    def get_average_of_knn_distance(self, indices_x, indices_y, nearest_k):
        """Get the average k-th NN distance within samples
        is equivalent to DistanceMatrix.get_knn_radius(..., nearest_k+1).mean()"""
        indices = np.argpartition(self(indices_x, indices_y), nearest_k + 1, axis=-1)[..., : nearest_k + 1]
        k_smallests = np.take_along_axis(self(indices_x, indices_y), indices, axis=-1)
        kth_values = k_smallests.max(axis=-1)
        k_nearest_radius = np.mean(kth_values)
        return k_nearest_radius

    def serialize(self):
        return {"device": self.device,
                "approx": self.approx}


def compute_decision(score_real, score_fake, gamma):
    """
    Mitigate the extreme values where P or Q are favoured
    Decision is taken as gamma*score_real>score_fake
    """
    if gamma == np.inf:
        return score_real > 0
    elif gamma == 0:
        return score_fake == 0
    if gamma < 1:
        return (gamma * score_real) > score_fake
    else:
        return (gamma * score_real) >= score_fake


def post_process_decisions(decisions, lambdas, gammas):
    extended_decisions = repeat_decisions_over_lambdas(decisions, nb_lambdas=len(lambdas))
    extended_decisions = enforce_extreme_gammas_for_extreme_lambdas(extended_decisions, lambdas, gammas)
    return extended_decisions


def repeat_decisions_over_lambdas(decisions, nb_lambdas):
    return np.repeat(decisions[..., np.newaxis], nb_lambdas, axis=-1)


def enforce_extreme_gammas_for_extreme_lambdas(decisions, lambdas, gammas):
    decisions = enforce_same_gamma_as_lambda(decisions, lambdas, gammas, ref=np.inf)
    decisions = enforce_same_gamma_as_lambda(decisions, lambdas, gammas, ref=0)
    return decisions


def enforce_same_gamma_as_lambda(decisions, lambdas, gammas, ref):
    def first_index_where(condition):
        indices, = np.where(condition)
        return indices[0]

    ind_gamma = first_index_where(gammas == ref)
    ind_lambda = first_index_where(lambdas == ref)
    decisions[..., ind_lambda] = decisions[ind_gamma, :, ind_lambda]
    return decisions


def compute_pairwise_distance(data_x, data_y=None, device="cpu", approx=False):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([M, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, M], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    compute_mode = "use_mm_for_euclid_dist_if_necessary" if approx else "donot_use_mm_for_euclid_dist"
    data_x = torch.tensor(data_x, device=device).unsqueeze(0)
    data_y = torch.tensor(data_y, device=device).unsqueeze(0)
    dists = torch.cdist(data_x, data_y, compute_mode=compute_mode).squeeze(0)
    return dists.cpu().numpy()


def compute_pairwise_distance_batched(data_x, data_y=None, device="cpu", approx=False,
                                      batch_size=BATCH_SIZE_DIST_MATRIX):
    """
    Variation of compute_pairwise_distance in which the computation is batched in order to reduce memory
    usage in the torch.cdist function which computes large intermediate matrices.
    In this method, the batching is made on the dimension corresponding the largest data object
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([M, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, M], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    compute_mode = "use_mm_for_euclid_dist_if_necessary" if approx else "donot_use_mm_for_euclid_dist"
    if isinstance(data_x, np.ndarray):
        data_x = torch.tensor(data_x, device=device)
    if isinstance(data_y, np.ndarray):
        data_y = torch.tensor(data_y, device=device)
    data_x = data_x.to(device)
    data_y = data_y.to(device)
    dist_matrix = torch.zeros((data_x.shape[0], data_y.shape[0]), device=device)
    batch_axis = "x" if data_x.shape[0] > data_y.shape[0] else "y"
    data_dict = {"x": data_x, "y": data_y}
    number_batches = ceil(data_dict[batch_axis].shape[0] / batch_size)
    for batch_index in range(number_batches):
        if batch_axis == "x":
            batch_distance_matrix = torch.cdist(data_x[(batch_index) * batch_size:(batch_index + 1) * batch_size],
                                                data_y, compute_mode=compute_mode)
            dist_matrix[batch_index * batch_size:(batch_index + 1) * batch_size, :] = batch_distance_matrix
        else:
            batch_distance_matrix = torch.cdist(data_x,
                                                data_y[(batch_index) * batch_size:(batch_index + 1) * batch_size],
                                                compute_mode=compute_mode)
            dist_matrix[:, batch_index * batch_size:(batch_index + 1) * batch_size] = batch_distance_matrix
    return dist_matrix.cpu().numpy()


def get_prd(lambdas, real_samples, fake_samples, classifiers):
    """
    Generic function which will run to get the precision recall couples for 2 distributions
    This is where the dual approach is used to compute the (alpha,beta) couples
        decisions_classifiers.shape = (n_gammas,nb_points_all_sample,n_lambdas)
    """
    labels_array = np.concatenate([np.repeat(1, real_samples.shape[0]),
                                   np.repeat(0, fake_samples.shape[0])])

    all_samples = np.concatenate([real_samples, fake_samples], axis=0)
    decisions_classifiers = classifiers(lambdas, all_samples)
    del all_samples
    del fake_samples, real_samples
    gc.collect()
    # adding classifiers f(z)=1 and f(z)=0
    decisions_classifiers = np.concatenate([np.zeros((1, decisions_classifiers.shape[1], decisions_classifiers.shape[2])),
                                            decisions_classifiers,
                                            np.ones((1, decisions_classifiers.shape[1], decisions_classifiers.shape[2]))],
                                           axis=0)

    fpr, fnr = compute_fpr_fnr(labels_array, decisions_classifiers)

    alphas_betas = compute_alphas_betas(lambdas=lambdas, fpr=fpr, fnr=fnr)

    return alphas_betas


def lazy_multiply(x, y):
    """Prevents returning nan values for extreme \lambda values"""
    x_masked = np.ma.masked_where(x == 0, x)
    y_masked = np.ma.masked_where(y == 0, y)
    result = x_masked * y_masked
    return np.ma.filled(result, 0)


def lazy_divide(x, y):
    y_masked = np.ma.masked_where(y == 0, y)
    y_inv = 1 / y_masked
    y_inv = np.ma.filled(y_inv, np.inf)
    return lazy_multiply(x, y_inv)


def compute_alphas_betas(lambdas, fpr, fnr):
    """Using dual approach from (Simon et al, 2018)"""
    lamb = lambdas[np.newaxis, :]

    alphas = lazy_multiply(lamb, fpr) + fnr
    betas = lazy_divide(fnr, lamb) + fpr

    alpha = np.min(alphas, axis=0)
    beta = np.min(betas, axis=0)

    alphas_betas = np.empty((lamb.shape[1], 2))
    alphas_betas[:, 0] = alpha
    alphas_betas[:, 1] = beta

    return alphas_betas


class GroundTruthClassifierEnsemble():
    def __init__(self, real_distrib, fake_distrib, samples) -> None:
        self.samples_gt = samples

        def classifier(lamb, all_samples):
            if lamb == np.inf:
                decisions = (real_distrib.eval_log_pdf(self.samples_gt[all_samples]) != -np.inf)
            elif lamb == 0:
                decisions = (fake_distrib.eval_log_pdf(self.samples_gt[all_samples]) == -np.inf)
            elif lamb < 1:
                decisions = (
                        (np.log(lamb) + real_distrib.eval_log_pdf(
                            self.samples_gt[all_samples])) > fake_distrib.eval_log_pdf(self.samples_gt[all_samples])
                ).astype(int)
            else:
                decisions = (
                        (np.log(lamb) + real_distrib.eval_log_pdf(
                            self.samples_gt[all_samples])) >= fake_distrib.eval_log_pdf(self.samples_gt[all_samples])
                ).astype(int)
            return decisions

        self.classifier = classifier

    def __call__(self, lambdas, all_samples) -> Any:
        """Call on all samples and return an array where decisions of shape [1,#samples,#lambdas] --> this will be broadcasted as necessary in get_prd"""
        decisions = np.array([self.classifier(lamb, all_samples) for lamb in lambdas]).T[np.newaxis, :, :]
        return decisions


def get_ground_truth_classifiers(real_distrib, fake_distrib, samples):
    """Return likelihood ratio classifier"""
    return GroundTruthClassifierEnsemble(real_distrib, fake_distrib, samples)


class KNNClassifierEnsemble():
    def __init__(self, gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix) -> None:
        self.gammas = gammas
        self.real_samples_train = real_samples_train
        self.fake_samples_train = fake_samples_train
        self.nearest_k = nearest_k
        # concatenate as the KNN is computed over the union of fake and real samples
        self.real_and_fake_train = np.append(real_samples_train, fake_samples_train, axis=0)

        self.distance_matrix = distance_matrix

    def __call__(self, lambdas, all_samples) -> Any:
        dists_test_to_train = self.distance_matrix(all_samples, self.real_and_fake_train)

        # radius from any test point to the k-th most distant point in the training set
        knn_test_in_train = self.distance_matrix.get_knn_radius(all_samples, self.real_and_fake_train,
                                                                nearest_k=self.nearest_k)

        score_real = (
                dists_test_to_train[:, : self.real_samples_train.shape[0]]
                <= knn_test_in_train[:, np.newaxis]
        ).sum(axis=1)

        score_fake = (
                dists_test_to_train[:, self.real_samples_train.shape[0]:]
                <= knn_test_in_train[:, np.newaxis]
        ).sum(axis=1)

        decisions = np.array([compute_decision(score_real, score_fake, gamma) for gamma in self.gammas])
        decisions = post_process_decisions(decisions, lambdas, self.gammas)
        return decisions


def get_knn_classifiers(real_samples_train, fake_samples_train, nearest_k, gammas, distance_matrix):
    """
    Output the parametrised family function of knn classifiers (improved IPR)
    """
    return KNNClassifierEnsemble(gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix)


class CoverageClassifierEnsemble():
    def __init__(self, gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix) -> None:
        self.gammas = gammas
        self.real_samples_train = real_samples_train
        self.fake_samples_train = fake_samples_train
        self.nearest_k = nearest_k
        self.distance_matrix = distance_matrix

    def __call__(self, lambdas, all_samples) -> Any:
        dists_test_to_train_real = self.distance_matrix(
            all_samples, self.real_samples_train
        )
        dists_test_to_train_fake = self.distance_matrix(
            all_samples, self.fake_samples_train
        )

        knn_test_in_train_real = self.distance_matrix.get_knn_radius(all_samples, self.real_samples_train,
                                                                     nearest_k=self.nearest_k)

        knn_test_in_train_fake = self.distance_matrix.get_knn_radius(all_samples, self.fake_samples_train,
                                                                     nearest_k=self.nearest_k)

        score_real = (
                dists_test_to_train_real[:, : self.real_samples_train.shape[0]]
                <= knn_test_in_train_fake[:, np.newaxis]
        ).sum(axis=1)

        score_fake = (
                dists_test_to_train_fake[:, : self.fake_samples_train.shape[0]]
                <= knn_test_in_train_real[:, np.newaxis]
        ).sum(axis=1)

        decisions = np.array([compute_decision(score_real, score_fake, gamma) for gamma in self.gammas])
        decisions = post_process_decisions(decisions, lambdas, self.gammas)

        return decisions


def get_coverage_classifiers(real_samples_train, fake_samples_train, nearest_k, gammas, distance_matrix):
    """
    Output the parametrised family function of knn classifiers (improved IPR)
    """
    return CoverageClassifierEnsemble(gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix)


class IPRClassifierEnsemble():
    def __init__(self, gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix) -> None:
        self.gammas = gammas
        self.real_samples_train = real_samples_train
        self.fake_samples_train = fake_samples_train

        self.distance_matrix = distance_matrix

        # compute the radiuses of KNN balls within X then within Y
        self.real_nearest_neighbour_distances = distance_matrix.get_knn_radius(real_samples_train, real_samples_train,
                                                                               nearest_k)
        self.fake_nearest_neighbour_distances = distance_matrix.get_knn_radius(fake_samples_train, fake_samples_train,
                                                                               nearest_k)

    def __call__(self, lambdas, all_samples):
        pairwise_distances_real = self.distance_matrix(all_samples, self.real_samples_train)
        pairwise_distances_fake = self.distance_matrix(all_samples, self.fake_samples_train)

        # compute the decisions of KNN classifiers on test data
        score_real = (
                pairwise_distances_real <= self.real_nearest_neighbour_distances[np.newaxis, ...]
        ).sum(axis=1)
        score_fake = (
                pairwise_distances_fake <= self.fake_nearest_neighbour_distances[np.newaxis, ...]
        ).sum(axis=1)

        decisions = np.array([compute_decision(score_real, score_fake, gamma) for gamma in self.gammas])
        decisions = post_process_decisions(decisions, lambdas, self.gammas)
        return decisions


def get_ipr_classifiers(real_samples_train, fake_samples_train, nearest_k, gammas, distance_matrix):
    """
    Get all the classifiers instanciated for the IPR classifiers class
    """
    return IPRClassifierEnsemble(gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix)


class PPRClassifierEnsemble():
    """NOT IN USE FOR THE MOMENT"""

    def __init__(self, gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix) -> None:
        self.gammas = list(gammas)
        self.real_samples_train = real_samples_train
        self.fake_samples_train = fake_samples_train

        self.distance_matrix = distance_matrix

        self.SCALING_FACTOR = 1.2

        self.pairwise_distance_between_real_train = distance_matrix(real_samples_train, real_samples_train)
        self.pairwise_distance_between_fake_train = distance_matrix(fake_samples_train, fake_samples_train)
        self.pairwise_distance_between_real_fake_train = distance_matrix(real_samples_train, fake_samples_train)
        self.pairwise_distance_between_fake_real_train = distance_matrix(fake_samples_train, real_samples_train)

        self.k_nearest_radius_x = distance_matrix.get_average_of_knn_distance(real_samples_train, real_samples_train,
                                                                              nearest_k=nearest_k)
        self.k_nearest_radius_y = distance_matrix.get_average_of_knn_distance(fake_samples_train, fake_samples_train,
                                                                              nearest_k=nearest_k)

        # store distance of the first neighbours in the real dist and in the fake dist to use as a relative 
        # comparison in terms of magnitude to correct matrix multiplication approximation in torch.cdist
        self.first_nearest_radius_x = distance_matrix.get_average_of_knn_distance(real_samples_train,
                                                                                  real_samples_train, nearest_k=1)
        self.first_nearest_radius_y = distance_matrix.get_average_of_knn_distance(fake_samples_train,
                                                                                  fake_samples_train, nearest_k=1)

    def __call__(self, lambdas, all_samples):
        # return decision from all_samples: evaluate the probabistic scoring rule on all_samples
        # decision will not necessarily (even unkikely) be 1 or 0 but will be the probability that sample z belongs to distribution P (real)
        # 2 implementations:
        # - either compute the estimation that z \in s_P(X) + z \in s_P(Y) then compute the decision as decsions = ((gamma*{z \in s_P(X)})>={z \in s_P(Y)}).astype(int)
        # - compute the decision solely on the function output {z \in s_P(X)} --> not depending on lamb

        pairwise_distances_real = self.distance_matrix(all_samples, self.real_samples_train)
        pairwise_distances_fake = self.distance_matrix(all_samples, self.fake_samples_train)

        decisions_real = get_scoring_rule_psr(distance=pairwise_distances_real.T,
                                              k_nearest_radius=self.k_nearest_radius_x,
                                              scaling_factor=self.SCALING_FACTOR)
        decisions_fake = 1 - get_scoring_rule_psr(distance=pairwise_distances_fake.T,
                                                  k_nearest_radius=self.k_nearest_radius_y,
                                                  scaling_factor=self.SCALING_FACTOR)

        # choose power version or convex combination
        decisions = np.array(
            [interpolate(decisions_real, decisions_fake, gamma / (gamma + 1)) for gamma in self.gammas])

        decisions = np.concatenate([decisions, [decisions_real, decisions_fake]], axis=0)
        decisions = post_process_decisions(decisions, lambdas, self.gammas)

        return decisions


def get_ppr_classifiers(real_samples_train, fake_samples_train, nearest_k, gammas, distance_matrix):
    """
    Probabilistic classifiers from Park, Dogyun, and Suhyun Kim. Probabilistic Precision and Recall
    Classifier is probabilistic in the sense that contrary to IPR and Diversity and Coverage, the manifold is not approximated
    by a union or intersection of ball with discontinuous boundaries
    
    The classifier is determined by f(z)=PSR_P (z) = 1 − product_{x \in X}(1 − Pr(z \in s_P (x))) = Pr(z ∈ S_P)

    """
    return PPRClassifierEnsemble(gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix)


class ParzenClassifierEnsemble():
    """
    p(z)=(1/N)\Sigma{sur N} kernel_i(z-x_i))
    """

    def __init__(self, gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix,
                 adaptative=False) -> None:
        self.gammas = gammas
        self.real_samples_train = real_samples_train
        self.fake_samples_train = fake_samples_train
        self.nearest_k = nearest_k
        self.SCALING_FACTOR = 1.

        self.distance_matrix = distance_matrix

        self.k_nearest_radius_x = distance_matrix.get_average_of_knn_distance(real_samples_train, real_samples_train,
                                                                              nearest_k=nearest_k)
        self.k_nearest_radius_y = distance_matrix.get_average_of_knn_distance(fake_samples_train, fake_samples_train,
                                                                              nearest_k=nearest_k)

        if adaptative:
            self.k_nearest_radius_x = distance_matrix.get_knn_radius(
                real_samples_train, real_samples_train, nearest_k=self.nearest_k
            )[:, np.newaxis]

            self.k_nearest_radius_y = distance_matrix.get_knn_radius(
                fake_samples_train, fake_samples_train, nearest_k=self.nearest_k
            )[:, np.newaxis]

        # store distance of the first neighbours in the real dist and in the fake dist to use as a relative
        # comparison in terms of magnitude to correct matrix multiplication approximation in torch.cdist
        self.first_nearest_radius_x = distance_matrix.get_average_of_knn_distance(real_samples_train,
                                                                                  real_samples_train, nearest_k=1)
        self.first_nearest_radius_y = distance_matrix.get_average_of_knn_distance(fake_samples_train,
                                                                                  fake_samples_train, nearest_k=1)

    def __call__(self, lambdas, all_samples):
        # return decision from all_samples: evaluate the probabistic scoring rule on all_samples
        # decision will not necessarily (even unlikely) be 1 or 0 but will be the probability that sample z belongs to distribution P (real)
        # 2 implementations:
        # - either compute the estimation that {z \in s_P(X)} and {z \in s_P(Y)} then compute the decision 
        #       as decisions = ((gamma*{z \in s_P(X)})>={z \in s_P(Y)}).astype(int)
        # - compute the decision solely on the function output {z \in s_P(X)} --> not depending on lamb

        pairwise_distances_real = self.distance_matrix(all_samples, self.real_samples_train)
        pairwise_distances_fake = self.distance_matrix(all_samples, self.fake_samples_train)

        score_real = get_parzen_score(distance=pairwise_distances_real.T, k_nearest_radius=self.k_nearest_radius_x,
                                      scaling_factor=self.SCALING_FACTOR)
        score_fake = get_parzen_score(distance=pairwise_distances_fake.T, k_nearest_radius=self.k_nearest_radius_y,
                                      scaling_factor=self.SCALING_FACTOR)

        # choose power version or convex combination
        decisions = np.array([compute_decision(score_real, score_fake, gamma) for gamma in self.gammas])
        decisions = post_process_decisions(decisions, lambdas, self.gammas)

        return decisions


def get_parzen_classifiers(real_samples_train, fake_samples_train, nearest_k, gammas, distance_matrix):
    return ParzenClassifierEnsemble(gammas, real_samples_train, fake_samples_train, nearest_k, distance_matrix)


def get_parzen_score(distance, k_nearest_radius, scaling_factor, gpu=False):
    # distance.shape(Dist of support)
    # out_of_knearest = distance >= SCALING_FACTOR * k_nearest_radius
    # parzen_scores = 1 - distance / (SCALING_FACTOR * k_nearest_radius)
    # parzen_scores[out_of_knearest] = 0.0

    parzen_scores = (distance <= k_nearest_radius).astype(int).mean(axis=0)
    return parzen_scores


def interpolate(fp, fq, pi):
    # NOTE case pi < 0.5
    # pi = gamma/(gamma+1)
    # return 0.5+np.sign(fp-0.5)*np.power(np.abs(fp-0.5),pi)*np.sign(fq-0.5)*np.power(np.abs(fq-0.5),1-pi)
    if pi <= 0.5:
        return np.power(fp, pi) * np.power(fq, 1 - pi)
    else:
        return 1 - np.power(1 - fp, pi) * np.power(1 - fq, 1 - pi)


def get_scoring_rule_psr(distance, k_nearest_radius, scaling_factor):
    # input matrix of size (n,m) n=samples of the support, m=samples to compare
    # import pdb; pdb.set_trace()
    # NOTE: in the original code, psr = np.prod(1.0 - psr, axis = 0) which is not in accordance to the article
    # definition
    # out_of_knearest = distance >= scaling_factor * k_nearest_radius
    psr = np.maximum(1 - distance / (scaling_factor * k_nearest_radius), 0)
    # psr[out_of_knearest] = 0.0
    # psr = 1 - np.exp(np.sum(np.log(1.0 - psr), axis = 0))
    psr = 1 - np.prod(1.0 - psr, axis=0)
    return psr
