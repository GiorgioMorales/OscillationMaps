"""
prdc 
Copyright (c) 2020-present NAVER Corp.
MIT license
Taken from Naeem et al: https://github.com/clovaai/generative-evaluation-prdc/
Updated the test (distance to nearest neighbour) < (knn radius)
to be lower or equal so that in the discrete setting, a point is indeed classified as belonging to
its own distribution
"""
import numpy as np
import sklearn.metrics

__all__ = ["compute_prdc", "get_kth_value"]


def compute_pairwise_distance(data_x, data_y=None):
    """
    Args:
        data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
        data_y: numpy.ndarray([M, feature_dim], dtype=np.float32)
    Returns:
        numpy.ndarray([N, M], dtype=np.float32) of pairwise distances.
    """
    if data_y is None:
        data_y = data_x
    dists = sklearn.metrics.pairwise_distances(
        data_x, data_y, metric="euclidean", n_jobs=8
    )
    return dists


def get_kth_value_batched(unsorted, k, batch_size=1000, axis=-1):
    pass


def get_kth_value(unsorted, k, axis=-1):
    """
    MODIFIED TO RUN FASTER
    Args:
        unsorted: numpy.ndarray of any dimensionality.
        k: int
    Returns:
        kth values along the designated axis.
    """
    # indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    # k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    # kth_values = k_smallests.max(axis=axis)
    kth_values = np.partition(unsorted, k - 1, axis=axis)[..., k - 1]
    return kth_values


def compute_nearest_neighbour_distances(input_features, nearest_k, split=False):
    """
    Args:
        input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int
        split: wether the dataset is splitted into train/test
    Returns:
        Distances to kth nearest neighbours.
    """
    k_val = nearest_k
    distances = compute_pairwise_distance(input_features)
    radii = get_kth_value(distances, k=k_val, axis=-1)
    return radii


def compute_nearest_neighbour_distances_from_dist_matrix(distances, nearest_k):
    """
    Compute nearest neighbour from any distance matrix.
    If symmetric matrix <-> the initial compute_nearest_neighbour...
    If not symmetric or not even square : computes the kNN radius centered in the values
        from the rows [distances.shape[0]] to its kth nearest neighbour in the colums (distances.shape[1])
    Args:
        distances: numpy.ndarray([N0, N1], dtype=np.float32)
        nearest_k: int
    Returns:
        Distances to kth nearest neighbours from values of N0 to those of N1
    """
    radii = get_kth_value(distances, k=nearest_k, axis=-1)
    return radii


def compute_prdc(real_features, fake_features, nearest_k):
    """
    Computes precision, recall, density, and coverage given two manifolds.

    Args:
        real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
        nearest_k: int.
    Returns:
        dict of precision, recall, density, and coverage.
    """
    real_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        real_features, nearest_k
    )
    fake_nearest_neighbour_distances = compute_nearest_neighbour_distances(
        fake_features, nearest_k
    )
    distance_real_fake = compute_pairwise_distance(real_features, fake_features)

    precision = (
        (distance_real_fake <= np.expand_dims(real_nearest_neighbour_distances, axis=1))
        .any(axis=0)
        .mean()
    )

    recall = (
        (distance_real_fake <= np.expand_dims(fake_nearest_neighbour_distances, axis=0))
        .any(axis=1)
        .mean()
    )

    density = (1.0 / float(nearest_k)) * (
            distance_real_fake <= np.expand_dims(real_nearest_neighbour_distances, axis=1)
    ).sum(axis=0).mean()

    coverage = (
            distance_real_fake.min(axis=1) <= real_nearest_neighbour_distances
    ).mean()
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)
