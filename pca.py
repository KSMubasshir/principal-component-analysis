# -*- coding: utf-8 -*-
# Import required python modules
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import scipy.stats as sp


def plot_clusters(prob_arr, data_points, means, num_of_components, num_of_data_points):
    for i in range(0, num_of_data_points):
        max_ind = np.argmax(prob_arr[i, :])
        if max_ind == 0:
            plt.scatter(data_points[i, 0], data_points[i, 1], facecolors='none', edgecolors='r')
        elif max_ind == 1:
            plt.scatter(data_points[i, 0], data_points[i, 1], facecolors='none', edgecolors='g')
        else:
            plt.scatter(data_points[i, 0], data_points[i, 1], facecolors='none', edgecolors='b')
    for k in range(0, num_of_components):
        plt.scatter(means[k, 0], means[k, 1], color='yellow', )
    plt.savefig('images/pca_clustered.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_principal_components(data):
    plt.scatter(data[:, 0], data[:, 1], s=80, facecolors='none', edgecolors='b')
    plt.savefig('images/pca_basic.png', dpi=300, bbox_inches='tight')
    plt.show()


def principal_component_analysis():
    # Load Digits dataset
    x, y = datasets.load_digits(return_X_y=True)
    x_std = StandardScaler().fit_transform(x)
    mean_vec = np.mean(x_std, axis=0)
    cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0] - 1)
    eig_values, eig_vectors = np.linalg.eig(cov_mat)
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_values[i]), eig_vectors[:, i]) for i in range(len(eig_values))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(key=lambda z: z[0], reverse=True)
    matrix_w = np.hstack((eig_pairs[0][1].reshape(-1, 1),
                          eig_pairs[1][1].reshape(-1, 1)))
    converted_input = np.dot(x_std, matrix_w)
    return converted_input


class GaussianMixModel(object):
    def __init__(self, x, k=2):
        # Algorithm can work for any number of columns in dataset
        x = np.asarray(x)
        self.m, self.n = x.shape
        self.data = x.copy()
        # number of mixtures
        self.k = k

    def _init(self):
        # init mixture means/sigmas
        np.random.seed(41)
        self.mean_arr = np.asmatrix(np.random.random((self.k, self.n)) + np.mean(self.data))
        self.sigma_arr = np.array([np.asmatrix(np.identity(self.n)) for _ in range(self.k)])
        self.phi = np.ones(self.k) / self.k
        self.Z = np.asmatrix(np.empty((self.m, self.k), dtype=float))
        # Z Latent Variable giving probability of each point for each distribution

    def fit(self, tol=1e-4):
        # Algorithm will run until max of log-likelihood is achieved
        self._init()
        number_of_iterations = 0
        log_likelihood = 1
        previous_log_likelihood = 0
        print('Initial')
        print("Means: " + str(self.mean_arr))
        print("Covariances: " + str(self.sigma_arr))
        print("Mixing Coefficients: " + str(self.phi) + "\n")

        while log_likelihood - previous_log_likelihood > tol:
            previous_log_likelihood = self.log_likelihood()
            self.e_step()
            self.m_step()
            number_of_iterations += 1
            log_likelihood = self.log_likelihood()

        print('Final')
        print("Means: " + str(self.mean_arr))
        print("Covariances: " + str(self.sigma_arr))
        print("Mixing Coefficients: " + str(self.phi) + "\n")
        print("Num of iterations: " + str(number_of_iterations))

        with open("probabilities.txt", "w") as prob_file:
            prob_file.write((str(self.Z)))

    def log_likelihood(self):
        log_likelihood = 0
        for i in range(self.m):
            tmp = 0
            for j in range(self.k):
                tmp += sp.multivariate_normal.pdf(self.data[i, :], self.mean_arr[j, :].A1, self.sigma_arr[j, :]) * \
                       self.phi[j]
            log_likelihood += np.log(tmp)
        return log_likelihood

    def e_step(self):
        # Finding probability of each point belonging to each pdf and putting it in latent variable Z
        for i in range(self.m):
            den = 0
            for j in range(self.k):
                num = sp.multivariate_normal.pdf(self.data[i, :],
                                                 self.mean_arr[j].A1,
                                                 self.sigma_arr[j]) * \
                      self.phi[j]
                den += num
                self.Z[i, j] = num
            self.Z[i, :] /= den
            assert self.Z[i, :].sum() - 1 < 1e-4  # Program stop if this condition is false

    def m_step(self):
        # Updating mean and variance
        for j in range(self.k):
            const = self.Z[:, j].sum()
            self.phi[j] = 1 / self.m * const
            _mu_j = np.zeros(self.n)
            _sigma_j = np.zeros((self.n, self.n))
            for i in range(self.m):
                _mu_j += (self.data[i, :] * self.Z[i, j])
                _sigma_j += self.Z[i, j] * (
                            (self.data[i, :] - self.mean_arr[j, :]).T * (self.data[i, :] - self.mean_arr[j, :]))
            self.mean_arr[j] = _mu_j / const
            self.sigma_arr[j] = _sigma_j / const


def main(no_of_components):
    pc = principal_component_analysis()
    plot_principal_components(pc)
    gmm = GaussianMixModel(pc, no_of_components)
    gmm.fit()
    plot_clusters(gmm.Z, gmm.data, gmm.mean_arr, gmm.k, 1000)


if __name__ == "__main__":
    main(int(sys.argv[1]))
