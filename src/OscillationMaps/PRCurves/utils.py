"""Helper functions used in the computation of precision and recall curves"""

from sklearn.metrics import confusion_matrix
import numpy as np
import sys

DASHES = "#" * 12


def compute_fpr_fnr(true_labels, predicted_labels):
    """
    Output false positive rate and false negative rate based on true labels and predicted labels
    of a binary classifier
    
    NOTE positive and negative are "inverted" compared to standard classification in the hypothesis testing
    z is positive --> f(z)=0 --> predicted as z \sim P
    z is negative --> f(z)=1 --> predicted as z \sim Q
    false positive: x \in P but f(x)=1 i.e classified as x \in Q
    false positive rate: FP/(FP+TN)=FP/N = true_classes*(1-predicted_classes)/length
    (fpr, fnp).shape = (n_gammas, n_lambdas)
    Args:
        true_labels: <array like>
        predicted_labels: <array like>

    Returns:
        fpr: float false positive rate
        fnr: float false negative rate
    """
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    if not isinstance(predicted_labels, np.ndarray):
        predicted_labels = np.array(predicted_labels)

    true_labels = true_labels[np.newaxis, :, np.newaxis]

    if np.sum(true_labels) == 0:
        fpr = 0
    else:
        fpr = np.sum(true_labels * (1 - predicted_labels), axis=1) / (np.sum(true_labels))
    if np.sum(1 - true_labels) == 0:
        fnr = 0
    else:
        fnr = np.sum((1 - true_labels) * (predicted_labels), axis=1) / (np.sum(1 - true_labels))
    return fpr, fnr


def get_angles(num_angles=100, epsilon=1e-10):
    if not (0 < epsilon < 0.1):
        raise ValueError("epsilon must be in (0, 0.1] but is %s." % str(epsilon))
    if not (3 <= num_angles <= 1e6):
        raise ValueError("num_angles must be in [3, 1e6] but is %d." % num_angles)
    angles = np.linspace(epsilon, np.pi / 2 - epsilon, num=num_angles)
    return angles


def get_slopes(num_angles=100, epsilon=1e-10):
    """
    From Sajjadi
    Return the corresponding values of parameter lambda for creation of PR curve
    Modified to add 0 and + infty
    Args:
        num_angles: int number of values of lambdas discretise
        epsilon: minimum value to prevent division by 0

    Returns:
        slopes: array of values of lambda
    """

    angles = get_angles(num_angles, epsilon)
    # Compute slopes for linearly spaced angles between [0, pi/2]
    slopes = np.tan(angles)
    slopes = np.concatenate([[0], slopes, [np.inf]])

    return slopes


def get_arg_mins(array, axis=None):
    """
    Get a list of argmin values of array and not only the first value of argmins
    """
    min_val = np.min(array, axis=axis)
    return np.where(array == min_val)


def truncate_gaussian(samples, n0):
    """return truncation of gaussian : values are between -n0 and +n0

    Args:
        samples (<array like>): samples to modify
        n0 (<array like>): threshold at which to truncate

    Returns:
        list type: truncated gaussian
    """
    truncated_samples = np.where(np.abs(samples) <= n0, samples, n0 * np.sign(samples))
    return truncated_samples


def interpolate(alphas, betas, nb_points=None):
    """interpolate alphas and betas from the radial version of alphas and betas"""
    if not nb_points:
        nb_points = len(alphas)
    beta_interp = np.linspace(0, 1, num=nb_points)
    # NOTE: alphas and betas must be reverted, as x axis is the beta which is decreasing
    # as it is \lambda for lamda range in (0,+\infty) goes from right to left
    alpha_interp = np.interp(beta_interp, betas[::-1], alphas[::-1])
    return alpha_interp, beta_interp


def intersection(alphas_1, alphas_2, x):
    assert len(alphas_1) == len(alphas_2)
    mini = np.minimum(alphas_1, alphas_2)
    return np.trapz(mini, x)


def union(alphas_1, alphas_2, x):
    assert len(alphas_1) == len(alphas_2)
    maxi = np.maximum(alphas_1, alphas_2)
    return np.trapz(maxi, x)


def iou_score(alphas_1, alphas_2, x):
    """
    intersection over union score
    make sure that len(alphas_1)==len(alphas_2)!
    If interpolation is used: give only the values of alphas
    x = must be the sale as beta_interp, i.e. np.linspace(0,1,nb_points)
    
    Usage example:
    nb_points = 111
    alphas_knn_interp,x = interpolate(alphas_knn,betas_knn,nb_points)
    alphas_gt_interp,_ = interpolate(alphas_gt,betas_gt,nb_points)
    score = iou_score(alphas_knn_interp,alphas_gt_interp,x)
    print(score)
    """
    return intersection(alphas_1, alphas_2, x) / union(alphas_1, alphas_2, x)


class PrintColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_print(message, color):
    print(f"{color}{message}{PrintColors.ENDC}")


def print_colored_text(text, color=PrintColors.BOLD):
    colored_print(f"\n\n{DASHES} {text} {DASHES}", color)


def print_to_file(message, file_name="/home/2024032/bsykes02/improved-ipr/tmp/output_log_file.txt"):
    with open(file_name, "a") as f:
        f.write(f"{message}\n")


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
