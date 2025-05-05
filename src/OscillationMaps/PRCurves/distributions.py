"""Define the distribution classes/functions which will allow to sample points and evaluate PDF"""

from scipy.stats import multivariate_normal
import numpy as np

IMPLEMENTED_VERSIONS = ["gaussian", "gaussian-mixture", "discrete"]


class Discrete(object):
    def __init__(self, masses, centroids) -> None:
        self.masses = np.array(masses)
        self.centroids = np.array(centroids, dtype=float)
        self.cum_masses = np.cumsum(self.masses)
        self.feature_dim = centroids.shape[1]

    def sample(self, size):
        probas = np.random.rand(size)
        selected_centroids = np.argmin(probas[:, np.newaxis] > self.cum_masses[np.newaxis, :], axis=-1)
        return self.centroids[selected_centroids]

    def eval_pdf(self, x):
        if x not in self.centroids:
            return 0
        else:
            return self.masses[np.where(x == self.centroids)]


class Gaussian(object):
    """
    Multivariate Gaussian object. Allows to sample and evaluate pdf.
    This version is isotropic in terms of covariange i.e. cov ∝ Identity
    """

    def __init__(
            self,
            params,
            feature_dim,
    ) -> None:
        self.feature_dim = feature_dim
        self.loc, self.scale = params
        # initialising the multivariate normal distribution objects from which to use either pdf
        # (probability density function or to sample from) if necessary split into train and test
        # scale**2 = variance = sigma**2
        if np.isscalar(self.loc):
            mean = np.repeat(self.loc, self.feature_dim)
        else:
            mean = self.loc
        self.var = multivariate_normal(
            mean=mean,
            cov=self.scale ** 2 * np.eye(self.feature_dim),
        )

    def eval_pdf(self, x):
        """
        simpler evaluation of the pdf as the cov matrix is diagonal and with same values
        yields overflow error
        """
        return 1 / (np.sqrt((2 * np.pi) ** self.feature_dim) * self.scale ** self.feature_dim) * \
               np.exp(-np.sum(((x - self.loc) / self.scale) ** 2, axis=1) / 2)

    def eval_log_pdf(self, x):
        """
        log pdf allows to circumvent overflow errors and is much faster then self.var.logpdf()
        """
        return -self.feature_dim / 2 * np.log(2 * np.pi) - self.feature_dim * np.log(self.scale) - (1 / 2) * np.sum(
            ((x - self.loc) / self.scale) ** 2, axis=1)

    def sample(self, size):
        """function to sample real points, sampling many points for ground truth PR curve"""
        features = self.var.rvs(size=size)
        if self.feature_dim == 1:
            features = features.reshape(-1, 1)
        return features


class NonIsotropicGaussian(object):
    """
    Multivariate Gaussian object. Allows to sample and evaluate pdf
    In this class, the variance is not necesseraly isotropic.
    As opposed to the `Gaussian` object, the parameter is covariance and not 
    isotropic scale (std)
    """

    def __init__(
            self,
            mu,
            cov,
    ) -> None:
        self.feature_dim = cov.shape[0]
        self.cov = cov
        self.mu = mu
        self.inv_cov = np.linalg.inv(self.cov)

        self.var = multivariate_normal(
            mean=mu,
            cov=self.cov,
        )
        self.det_cov = np.linalg.det(self.cov)

    def eval_pdf(self, x):
        """
        simpler evaluation of the pdf as the cov matrix is diagonal and with same values
        yields overflow error
        """
        return 1 / (np.sqrt(
            ((2 * np.pi) ** self.feature_dim) * self.det_cov
        )
        ) * \
               np.exp(-(1 / 2) * np.sum((x - self.mu) * ((x - self.mu).dot(self.inv_cov)), axis=1))

    def eval_log_pdf(self, x):
        """
        log pdf allows to circumvent overflow errors and is much faster then self.var.logpdf()
        """
        return -self.feature_dim / 2 * np.log(2 * np.pi) - (1 / 2) * np.log(self.det_cov) - (1 / 2) * np.sum(
            (x - self.mu) * ((x - self.mu).dot(self.inv_cov)), axis=1)

    def sample(self, size):
        """function to sample real points, sampling many points for ground truth PR curve"""
        features = self.var.rvs(size=size)
        if self.feature_dim == 1:
            features = features.reshape(-1, 1)
        return features


class GaussianMixture(object):
    """
    Multivariate Gaussian object. Allows to sample and evaluate pdf
    """

    def __init__(
            self,
            params,
            feature_dim,
    ) -> None:
        """Creates a Mixture of Gaussians distribution where the parameters are stored in list
        number of modes = len(params)
        for i-th mode: the mode is samples with probability \pi_i and the mode itself is a gaussian \sim N(\mu_i,\sigma_i)
        params = [(mu_i,sigma_i,pi_i) for i range(number of modes)]

        Args:
            params (list): list of parameters of the distributions
            feature_dim (int): dimension of the feature space
        """
        # normalise so that sum(p_i)=1
        sum_pi = sum(list(zip(*params))[2])
        self.params = [(mu, sigma, pi / sum_pi) for mu, sigma, pi in params]
        self.feature_dim = feature_dim
        self.pdf = lambda x: sum(pi * multivariate_normal(mean=np.ones(feature_dim) * mu,
                                                          cov=np.eye(feature_dim) * sigma).pdf(x) for mu, sigma, pi in
                                 self.params)
        self.log_pdf = lambda x: np.log(self.pdf(x))

    def eval_pdf(self, x):
        return self.pdf(x)

    def eval_log_pdf(self, x):
        return self.log_pdf(x)

    def sample(self, size):
        modes = np.random.choice(len(self.params), size=size, p=list(zip(*self.params))[2])
        # Generate samples for all modes at once using vectorized operations
        samples = np.array([multivariate_normal.rvs(mean=np.ones(self.feature_dim) * mu,
                                                    cov=np.eye(self.feature_dim) * sigma,
                                                    size=size)
                            for mu, sigma, pi in self.params
                            ])
        samples = samples.reshape(len(self.params), -1, self.feature_dim)

        # samples.shape = [len(self.params), size        , self.feature_dim]
        #               = [nb_modes        , samples_size, self.feature_dim] 
        return samples[modes, np.arange(size), :]


def create_distrib(distrib, params, feature_dim):
    if distrib == "gaussian":
        if np.isscalar(params[1]):
            # in this setting, params[1] is the std=scale=sqrt(variance)
            return Gaussian(params=params, feature_dim=feature_dim)
        else:
            # in this setting, params[1] is variance=scale**2
            return NonIsotropicGaussian(*params)
    elif distrib == "gaussian-mixture":
        return GaussianMixture(params=params, feature_dim=feature_dim)
    elif distrib == "discrete":
        return Discrete(masses=params[0], centroids=params[1])
    raise NotImplementedError(
        f"Use any distribution in {', '.join(IMPLEMENTED_VERSIONS)}. Received not implemented {distrib} distribution"
    )


def test():
    """Test the Gaussian object to check if the densities match the implementation in scipy"""

    ################ Parse arguments ###################
    from scipy.stats import multivariate_normal
    def parse_args():
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--params", type=float, required=True, nargs="*")
        parser.add_argument("--feature-dim", type=int, required=True)
        parser.add_argument("--number-points", "-n", type=int, required=True)
        parser.add_argument("--random-seed", type=int, default=None)
        args = parser.parse_args()
        return args

    args = parse_args()
    if args.random_seed:
        np.random.seed(args.random_seed)

    print(f"Running test comparison Gaussian object vs scipy implementation with {args=}")

    x_array = np.random.uniform(-100, 100, (args.number_points, args.feature_dim))

    gauss = Gaussian(args.params, feature_dim=args.feature_dim)
    rv = multivariate_normal(mean=args.params[0] * np.ones(args.feature_dim),
                             cov=np.eye(args.feature_dim) * args.params[1] ** 2)

    pdf_gauss_obj = gauss.eval_pdf(x_array)
    pdf_scipy = rv.pdf(x_array)
    print(f"PDF match : {np.isclose(pdf_gauss_obj, pdf_scipy).all()}")

    log_pdf_gauss_obj = gauss.eval_log_pdf(x_array)
    log_pdf_scipy = rv.logpdf(x_array)
    print(f"log PDF match : {np.isclose(log_pdf_gauss_obj, log_pdf_scipy).all()}")


if __name__ == "__main__":
    test()
