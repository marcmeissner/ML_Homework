import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
sns.set_style('whitegrid')

from scipy.stats import multivariate_normal

X = np.loadtxt('faithful.txt')
plt.figure(figsize=[6, 6])
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Eruptions (minutes)')
plt.ylabel('Waiting time (minutes)')
plt.show()


def normalize_data(X):
    """Normalize data such that it lies in range [0, 1] along every dimension.

    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix, each row represents a sample.

    Returns
    -------
    X_norm : np.array, shape [N, D]
        Normalized data matrix. 
    """
    N = len(X[:,0])
    D = len(X[0,:])
    X_norm = (X-np.min(X,axis=0))/(np.max(X,axis=0) - np.min(X,axis=0))
    return X_norm

plt.figure(figsize=[6, 6])
X_norm = normalize_data(X)
plt.scatter(X_norm[:, 0], X_norm[:, 1]);


def gmm_log_likelihood(X, means, covs, mixing_coefs):
    """Compute the log-likelihood of the data under current parameters setting.

    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix with samples as rows.
    means : np.array, shape [K, D]
        Means of the GMM (\mu in lecture notes).
    covs : np.array, shape [K, D, D]
        Covariance matrices of the GMM (\Sigma in lecuture notes).
    mixing_coefs : np.array, shape [K]
        Mixing proportions of the GMM (\pi in lecture notes).

    Returns
    -------
    log_likelihood : float
        log p(X | \mu, \Sigma, \pi) - Log-likelihood of the data under the given GMM.
    """
    log_likelihood = 0
    K = len(mixing_coefs)
    for i in range(np.size(X, 0)):
        log_likelihood += np.log(sum( \
            [mixing_coefs[k] * \
             multivariate_normal.pdf(X[i, :], means[k, :], covs[k, :, :]) for k in range(K)]))

    return log_likelihood


def e_step(X, means, covs, mixing_coefs):
    """Perform the E step.

    Compute the responsibilities.

    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix with samples as rows.
    means : np.array, shape [K, D]
        Means of the GMM (\mu in lecture notes).
    covs : np.array, shape [K, D, D]
        Covariance matrices of the GMM (\Sigma in lecuture notes).
    mixing_coefs : np.array, shape [K]
        Mixing proportions of the GMM (\pi in lecture notes).

    Returns
    -------
    responsibilities : np.array, shape [N, K]
        Cluster responsibilities for the given data.
    """

    responsibilities = np.zeros([np.size(X, 0), len(mixing_coefs)])
    K = len(mixing_coefs)
    N = np.size(X, 0)

    for n in range(N):
        denominator = sum([mixing_coefs[k] * multivariate_normal.pdf( \
            X[n, :], means[k, :], covs[k, :, :]) \
                           for k in range(K)])

        responsibilities[n, :] = [mixing_coefs[k] * multivariate_normal.pdf( \
            X[n, :], means[k, :], covs[k, :, :]) \
                                  for k in range(K)] / denominator

    return responsibilities


def m_step(X, responsibilities):
    """Update the parameters \theta of the GMM to maximize E[log p(X, Z | \theta)].

    Parameters
    ----------
    X : np.array, shape [N, D]
        Data matrix with samples as rows.
    responsibilities : np.array, shape [N, K]
        Cluster responsibilities for the given data.

    Returns
    -------
    means : np.array, shape [K, D]
        Means of the GMM (\mu in lecture notes).
    covs : np.array, shape [K, D, D]
        Covariance matrices of the GMM (\Sigma in lecuture notes).
    mixing_coefs : np.array, shape [K]
        Mixing proportions of the GMM (\pi in lecture notes).

    """

    N = np.size(responsibilities, 0)
    K = np.size(responsibilities, 1)
    D = np.size(X, 1)

    covs = np.zeros([K, D, D])

    N_k = np.array([sum(responsibilities[:, k]) for k in range(K)])
    mixing_coefs = N_k[:] / N
    # assign \mu_k
    means = np.array([1 / N_k[k] * sum([responsibilities[n, k] * X[n, :] for \
               n in range(N)]) for k in range(K)])
    # assign covs_k
    for k in range(K):
        covs[k, :, :] = 1 / N_k[k] * np.sum([responsibilities[n, k] * \
                np.outer((X[n, :] - means[k, :]), (X[n, :] - means[k, :])) for \
                n in range(N)], axis=0)

    return means, covs, mixing_coefs


def plot_gmm_2d(X, responsibilities, means, covs, mixing_coefs):
    """Visualize a mixture of 2 bivariate Gaussians.

    This is badly written code. Please don't write code like this.
    """
    plt.figure(figsize=[6, 6])
    palette = np.array(sns.color_palette('colorblind', n_colors=3))[[0, 2]]
    colors = responsibilities.dot(palette)
    # Plot the samples colored according to p(z|x)
    plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.5)
    # Plot locations of the means
    for ix, m in enumerate(means):
        plt.scatter(m[0], m[1], s=300, marker='X', c=palette[ix],
                    edgecolors='k', linewidths=1, )
    # Plot contours of the Gaussian
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 1, 50)
    xx, yy = np.meshgrid(x, y)
    for k in range(len(mixing_coefs)):
        zz = mlab.bivariate_normal(xx, yy, np.sqrt(covs[k][0, 0]),
                                   np.sqrt(covs[k][1, 1]),
                                   means[k][0], means[k][1], covs[k][0, 1])
        plt.contour(xx, yy, zz, 2, colors='k')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()

X_norm = normalize_data(X)
max_iters = 20

# Initialize the parameters
means = np.array([[0.2, 0.6], [0.8, 0.4]])
covs = np.array([0.5 * np.eye(2), 0.5 * np.eye(2)])
mixing_coefs = np.array([0.5, 0.5])

old_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)
responsibilities = e_step(X_norm, means, covs, mixing_coefs)
print('At initialization: log-likelihood = {0}'
      .format(old_log_likelihood))
plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)

# Perform the EM iteration
for i in range(max_iters):
    responsibilities = e_step(X_norm, means, covs, mixing_coefs)
    means, covs, mixing_coefs = m_step(X_norm, responsibilities)
    new_log_likelihood = gmm_log_likelihood(X_norm, means, covs, mixing_coefs)
    # Report & visualize the optimization progress
    print('Iteration {0}: log-likelihood = {1:.2f}, improvement = {2:.2f}'
          .format(i, new_log_likelihood, new_log_likelihood - old_log_likelihood))
    old_log_likelihood = new_log_likelihood
    plot_gmm_2d(X_norm, responsibilities, means, covs, mixing_coefs)

