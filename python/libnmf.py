""" NMF (Nonnegative Matrix Factorization) for Spectrum Imaging Data Analysis
"""
# Author: Motoki Shiga <shiga_m@gifu-u.ac.jp>
# License: MIT
#
#  Reference
#  [1] Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda,
#      Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
#      "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
#      Ultramicroscopy, Vol.170, p.43-59, 2016.
#


import numpy as np
from numpy import random
import numpy.linalg as lin
from scipy.special import gammaln
import matplotlib.pyplot as plt


class NMF(object):
    """Non-Negative Matrix Factorization (NMF)

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set all features
        are kept.
    reps : The number of initializations. (default: 3)
    max_itr : integer, default: 200
        Number of iterations to compute.
    random_state : integer seed, RandomState instance (default: 0)
        Random number generator seed control.
    Attributes
    ----------
    C_ : array, [#spatial data points, n_components]
        Non-negative components decomposed from data X.
    S_ : array, [#channels, n_components]
        Non-negative spectra decomposed from data X.
    obj_fun_ : array, [#iterations]
        Learning curve of reconstruction error (Mean Squared Error)
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> model = NMF(n_components=2)
    >>> model.fit(X)
    Training NMF model....
    1th iteration of NMF algorithm
    2th iteration of NMF algorithm
    3th iteration of NMF algorithm

    NMF(n_components=2, reps=3, max_itr=100, random_seed=0)
    >>> model.C_
    array([[ 0.        ,  0.40549951],
       [ 0.13374645,  0.40555886],
       [ 0.24076597,  0.48667235],
       [ 0.40131387,  0.4055646 ],
       [ 0.56186177,  0.32445684],
       [ 0.66888128,  0.40557034]])
    >>> model.S_
    array([[ 7.47464589,  2.46643616],
       [ 0.        ,  2.4657656 ]])

    References
    ----------
    [1] Cichocki, Andrzej, and P.H.A.N. Anh-Huy.
        “Fast local algorithms for large scale nonnegative matrix and tensor factorizations.”
        IEICE transactions on fundamentals of electronics, communications and computer sciences 92.3: 708-721, 2009.
    """

    # constructor
    def __init__(self, n_components, reps=3, max_itr=100, random_seed=0, flag_nonneg=True):
        self.n_components = n_components
        self.reps = reps
        self.max_itr = max_itr
        self.random_seed = random_seed
        self.flag_nonneg = flag_nonneg

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + \
              ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, num_xy=list(), channel_vals=list(), unit_name='Channel'):
        """
        Learn a NMF model for the data X.
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        num_xy: {array}, (#samples in x axis, #samples in x axis)
                or int, (#samples in x axis x #samples in x axis)
            The number of data points
        channel_vals: {array}
            The sequence of channel numbers, or unit values
        unit_name: strings
            The name of x axis of spectra
        Returns
        -------
        self
        """

        num_X_xy, num_X_ch = X.shape

        if type(num_xy)==int:
            self.num_xy = [num_xy]
        elif len(num_xy)==0:
            self.num_xy = num_X_xy
        else:
            self.num_xy = num_xy

        if len(channel_vals)>0:
            self.channel_vals = channel_vals
        else:
            self.channel_vals = np.arange(num_X_ch)
        self.unit_name = unit_name

        obj_best = np.array([np.inf])
        random.seed(self.random_seed)  # set the random seed
        print('Training NMF model....')
        for rep in range(self.reps):
            print(str(rep + 1) + 'th iteration of NMF algorithm')

            # initialization
            obj = np.zeros(self.max_itr)
            C = np.ones((num_X_xy, self.n_components))
            for j in range(self.n_components):
                C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]) + 1e-16)
            cj = np.sum(C, axis=1)
            i = np.random.choice(num_X_xy, self.n_components)
            S = X[i, :].T

            # main loop
            for itr in range(self.max_itr):
                # update S
                XC = X.T @ C
                C2 = C.T @ C
                for j in range(self.n_components):
                    S[:, j] = XC[:, j] - S @ C2[:, j] + C2[j, j] * S[:, j]
                    if self.flag_nonneg:
                        S[:, j] = (S[:, j] + np.abs(S[:, j])) / 2  # replace negative values with zeros

                # update C
                XS = X @ S
                S2 = S.T @ S
                for j in range(self.n_components):
                    cj = cj - C[:, j]
                    C[:, j] = XS[:, j] - C @ S2[:, j] + S2[j, j] * C[:, j]
                    C[:, j] = (C[:, j] + np.abs(C[:, j])) / 2  # replace negative values with zeros
                    C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]))  # normalize
                    cj = cj + C[:, j]

                # cost function
                X_est = C @ S.T  # reconstructed data matrix
                obj[itr] = lin.norm(X - X_est, ord='fro')**2 / X.size

                # check of convergence
                if (itr > 1) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr]
                    print('# updates: ' + str(itr))
                    break

            # choose the best result
            if obj_best[-1] > obj[-1]:
                obj_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()

        self.C_, self.S_, self.obj_fun_ = C_best, S_best, obj_best
        return self

    def imshow_component(self, figsize=list()):
        '''
        Plot spatial distributions of components

        Parameters
        ----------
        figsize: the vertical and horizontal size of the figure
        '''

        if (type(self.num_xy) != int) and (len(self.num_xy) == 2):
            if len(figsize) == 0:
                plt.figure()
            else:
                plt.figure(figsize=figsize)
            for k in range(self.C_.shape[1]):
                plt.subplot(100 + self.C_.shape[1] * 10 + k + 1)
                im = self.C_[:, k].reshape(self.num_xy)
                plt.imshow(im)
                plt.title('Component: ' + str(k + 1))
            plt.tight_layout()
            plt.show()
        else:
            self.plot_component(figsize)

    def plot_component(self, figsize=list()):
        '''
        Plot component intensities (data points vs intensities)
        Parameters
        ----------
        figsize: the vertical and horizontal size of the figure
        '''
        if len(figsize) == 0:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.C_.shape[1]):
            plt.plot(self.C_[:, k], label=str(k + 1))
        plt.xlim([0, self.C_.shape[0]])
        plt.xlabel('Spatial data point')
        plt.ylabel('Intensity')
        plt.title('Components')
        plt.legend()
        plt.show()

    def plot_spectra(self, figsize=list(), normalize=True):
        '''
        Plot spectra

        Parameters
        ----------
        figsize: the vertical and horizontal size of the figure
        normalize: Normalize each spectrum or NOT
        '''
        if len(figsize) == 0:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.S_.shape[1]):
            if normalize:
                self.S_[:, k] = self.S_[:, k] / (np.sqrt(self.S_[:, k].T @ self.S_[:, k]) + 1e-16)
            plt.plot(self.channel_vals, self.S_[:, k], label=str(k + 1))
        plt.xlabel(self.unit_name)
        plt.ylabel('Intensity')
        plt.xlim([self.channel_vals[0], self.channel_vals[-1]])
        plt.title('Spectra')
        plt.legend()
        plt.show()

    def plot_object_fun(self, figsize=list()):
        '''
        Plot learning curve (#iterations vs object function (error function))
        Parameters
        ----------
        figsize: the vertical and horizontal size of the figure
        '''
        if len(figsize) == 0:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        plt.plot(self.obj_fun_)
        plt.xlabel('Iterations')
        plt.xlim([0, len(self.obj_fun_)])
        plt.title('Object function')
        plt.show()

class NMF_SO(NMF):
    """Non-Negative Matrix Factorization with Soft orthogonality penalty (NMF-SO)

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set all features
        are kept.
    wo : weight of orthogonal penalty.
        The value should be between 0 and 1.
    reps : The number of initializations. (default: 3)
    max_itr : integer, default: 200
        Number of iterations to compute.
    random_state : integer seed, RandomState instance (default: 0)
        Random number generator seed control.
    Attributes
    ----------
    C_ : array, [#spatial data points, n_components]
        Non-negative components decomposed from data X.
    S_ : array, [#channels, n_components]
        Non-negative spectra decomposed from data X.
    obj_fun_ : array, [#iterations]
        Learning curve of reconstruction error (Mean Squared Error)
    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> model = NMF_SO(n_components=2, wo = 0.1)
    >>> model.fit(X)
    Training NMF with Soft Orthogonal constraint....
    1th iteration of NMF-SO algorithm
    2th iteration of NMF-SO algorithm
    3th iteration of NMF-SO algorithm

    NMF_SO(n_components=2, wo=0.1, reps=3, max_itr=100, random_seed=0)
    >>> model.C_
    array([[ 0.        ,  0.30547946],
       [ 0.        ,  0.51238139],
       [ 0.        ,  0.73899883],
       [ 0.33013316,  0.31309478],
       [ 0.60391616,  0.        ],
       [ 0.72546355,  0.        ]])
    >>> model.S_
    array([[ 8.28515563,  3.94337313],
       [ 1.34447182,  1.87880282]])

    References
    ----------
    Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda,
     Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
     "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
     Ultramicroscopy, Vol.170, p.43-59, 2016.
     doi: 10.1016/j.ultramic.2016.08.006
    """

    # constructor
    def __init__(self, n_components, wo=0.1, reps=3, max_itr=100, random_seed=0, flag_nonneg=True):
        self.n_components = n_components
        self.wo = wo
        self.reps = reps
        self.max_itr = max_itr
        self.random_seed = random_seed
        self.flag_nonneg = flag_nonneg

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) + ', wo=' + str(self.wo) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + \
              ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, num_xy=list(), channel_vals=list(), unit_name='Channel'):
        """
        Learn a NMF model for the data X.
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        num_xy: {array}, (#samples in x axis, #samples in x axis)
                or int, (#samples in x axis x #samples in x axis)
            The number of data points
        channel_vals: {array}
            The sequence of channel numbers, or unit values
        unit_name: strings
            The name of x axis of spectra
        Returns
        -------
        self
        """


        num_X_xy, num_X_ch = X.shape

        if type(num_xy)==int:
            self.num_xy = [num_xy]
        elif len(num_xy)==0:
            self.num_xy = num_X_xy
        else:
            self.num_xy = num_xy

        if len(channel_vals)>0:
            self.channel_vals = channel_vals
        else:
            self.channel_vals = np.arange(num_X_ch)
        self.unit_name = unit_name

        obj_best = np.array([np.inf])
        random.seed(self.random_seed)  # set the random seed
        print('Training NMF with Soft Orthogonal constraint....')
        for rep in range(self.reps):
            print(str(rep + 1) + 'th iteration of NMF-SO algorithm')

            # initialization
            obj = np.zeros(self.max_itr)
            C = np.ones((num_X_xy, self.n_components))
            for j in range(self.n_components):
                C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]) + 1e-16)
            cj = np.sum(C, axis=1)
            i = np.random.choice(num_X_xy, self.n_components)
            S = X[i, :].T

            # main loop
            for itr in range(self.max_itr):
                # update S
                XC = X.T @ C
                C2 = C.T @ C
                for j in range(self.n_components):
                    S[:, j] = XC[:, j] - S @ C2[:, j] + C2[j, j] * S[:, j]
                    if self.flag_nonneg:
                        S[:, j] = (S[:, j] + np.abs(S[:, j])) / 2  # replace negative values with zeros

                # update C
                XS = X @ S
                S2 = S.T @ S
                for j in range(self.n_components):
                    cj = cj - C[:, j]
                    C[:, j] = XS[:, j] - C @ S2[:, j] + S2[j, j] * C[:, j]
                    C[:, j] = C[:, j] - self.wo * (cj.T @ C[:, j]) / (cj.T @ cj) * cj
                    C[:, j] = (C[:, j] + np.abs(C[:, j])) / 2  # replace negative values with zeros
                    C[:, j] = C[:, j] / (np.sqrt(C[:, j].T @ C[:, j]))  # normalize
                    cj = cj + C[:, j]

                # cost function
                X_est = C @ S.T  # reconstructed data matrix
                obj[itr] = lin.norm(X - X_est, ord='fro')**2 / X.size

                # check of convergence
                if (itr > 1) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr]
                    print('# updates: ' + str(itr))
                    break

            # choose the best result
            if obj_best[-1] > obj[-1]:
                obj_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()

        self.C_, self.S_, self.obj_fun_ = C_best, S_best, obj_best
        return self

class NMF_ARD_SO(NMF_SO):
    """Non-Negative Matrix Factorization with Soft orthogonality penalty (NMF-SO)

        Parameters
        ----------
        n_components : int or None
            Number of components, if n_components is not set all features
            are kept.
        wo : real value
            The weight of orthogonal penalty.
            The value should be between 0 and 1.
        reps : The number of initializations. (default: 3)
        max_itr : integer, default: 200
            Number of iterations to compute.
        alpha: real value (over than 1)
            To adjust sparseness
        threshold_merge: real value
            The threshold of similarity between components to judge components should be merged.
        random_state : integer seed, RandomState instance (default: 0)
            Random number generator seed control.
        Attributes
        ----------
        C_ : array, [#spatial data points, n_components]
            Non-negative components decomposed from data X.
        S_ : array, [#channels, n_components]
            Non-negative spectra decomposed from data X.
        obj_fun_ : array, [#iterations]
            Learning curve of reconstruction error (Mean Squared Error)
        beta_ : real value
            Sparse penalty parameter (computed from alpha and data X)
        lambdas_ : attay, [#iterations]
            Learning curve of component intensities
        Examples
        --------
        >>> import numpy as np
        >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
        >>> model = NMF_ARD_SO(n_components=2, wo = 0.1)
        >>> model.fit(X)
        Training NMF with Soft Orthogonal constraint....
        1th iteration of NMF-SO algorithm
        2th iteration of NMF-SO algorithm
        3th iteration of NMF-SO algorithm

        NMF_SO(n_components=2, wo=0.1, reps=3, max_itr=100, random_seed=0)
        >>> model.C_
        array([[ 0.        ,  1.31254938],
           [ 0.        ,  2.21337851],
           [ 0.04655829,  3.15615036],
           [ 2.88446237,  1.23380528],
           [ 5.05090679,  0.        ],
           [ 6.07007114,  0.        ]])
        >>> model.S_
            array([[ 0.9869102 ,  0.90082913],
                   [ 0.16127074,  0.43417379]])

        References
        ----------
        Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda,
         Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
         "Sparse Modeling of EELS and EDX Spectral Imaging Data by Nonnegative Matrix Factorization",
         Ultramicroscopy, Vol.170, p.43-59, 2016.
         doi: 10.1016/j.ultramic.2016.08.006
    """

    # constructor
    def __init__(self, n_components, wo=0.1, reps=3, max_itr=100,
                 alpha=1+10**(-15), threshold_merge=0.99, random_seed=0, flag_nonneg=True):
        super(NMF_ARD_SO, self).__init__(n_components, wo, reps, max_itr, random_seed)
        self.alpha = alpha
        self.threshold_merge = threshold_merge
        self.flag_nonneg = flag_nonneg

    def __repr__(self):
        class_name = self.__class__.__name__
        txt = 'n_components=' + str(self.n_components) + ', wo=' + str(self.wo) \
              + ', reps=' + str(self.reps) + ', max_itr=' + str(self.max_itr) + \
              ', alpha=' + str(self.alpha) + ', threshold_merge=' + str(self.threshold_merge) + ', random_seed=' + str(self.random_seed)
        return '%s(%s)' % (class_name, txt,)

    def __str__(self):
        txt = self.__repr__()
        return txt

    def fit(self, X, num_xy=list(), channel_vals=list(), unit_name='Channel'):
        """
        Learn a NMF model for the data X.
        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed
        num_xy: {array}, (#samples in x axis, #samples in x axis)
                or int, (#samples in x axis x #samples in x axis)
            The number of data points
        channel_vals: {array}
            The sequence of channel numbers, or unit values
        unit_name: strings
            The name of x axis of spectra
        Returns
        -------
        self
        """

        eps = np.finfo(np.float64).eps  # tiny value
        num_X_xy, num_X_ch = X.shape  # the number of data points and the number of channels

        # --- Attribute initialization from a data matrix------
        if type(num_xy) == int:
            self.num_xy = [num_xy]
        elif len(num_xy) == 0:
            self.num_xy = num_X_xy
        else:
            self.num_xy = num_xy  # (xdim, ydim)

        if len(channel_vals) > 0:
            self.channel_vals = channel_vals
        else:
            self.channel_vals = np.arange(num_X_ch)
        self.unit_name = unit_name
        # -----------------------------------------------------

        mu_x = np.mean(X)
        self.beta_ = mu_x * (self.alpha - 1) * np.sqrt(num_X_ch) / self.n_components
        const = self.n_components * (gammaln(self.alpha) - self.alpha * np.log(self.beta_))
        random.seed(self.random_seed)  # set the random seed


        obj_best = np.array([np.inf])  # to deposit the best object value
        print('Training NMF with ARD and Soft Orthogonal constraint....')
        for rep in range(self.reps):
            print(str(rep+1) + 'th iteration of NMF-ARD-SO algorithm')

            # --- Initialization ------
            C = (np.random.rand(num_X_xy, self.n_components) + 1) * (np.sqrt(mu_x / self.n_components))
            L = (np.sum(C, axis=0) + self.beta_) / (num_X_ch + self.alpha + 1)
            cj = np.sum(C, axis=1)
            i = np.random.choice(num_X_xy, self.n_components)
            S = X[i, :].T
            for j in range(self.n_components):
                c = (np.sqrt(S[:, j].T @ S[:, j]))  # normalize
                if c > 0:
                    S[:, j] = S[:, j] / c
                else:
                    S[:, j] = 1 / np.sqrt(num_X_ch)
            X_est = C @ S.T  # reconstructed data matrix
            sigma2 = np.mean((X - X_est) ** 2)
            obj = np.zeros(self.max_itr)
            lambdas = np.zeros((self.max_itr, self.n_components))
            # -------------------------

            for itr in range(self.max_itr):
                # update S (spectra)
                XC = X.T @ C
                C2 = C.T @ C
                for j in range(self.n_components):
                    S[:, j] = XC[:, j] - S @ C2[:, j] + C2[j, j] * S[:, j]
                    if self.flag_nonneg:
                        S[:, j] = (S[:, j] + np.abs(S[:, j])) / 2  # replace negative values with zeros
                    c = (np.sqrt(S[:, j].T @ S[:, j]))  # normalize
                    if c > 0:
                        S[:, j] = S[:, j] / c
                    else:
                        S[:, j] = 1 / np.sqrt(num_X_ch)

                # update C (component intensities)
                XS = X @ S
                S2 = S.T @ S
                for j in range(self.n_components):
                    cj = cj - C[:, j]
                    C[:, j] = XS[:, j] - C @ S2[:, j] + S2[j, j] * C[:, j]
                    C[:, j] = C[:, j] - sigma2 / L[j]
                    if (self.wo > 0):
                        C[:, j] = C[:, j] - self.wo * (cj.T @ C[:, j]) / (cj.T @ cj) * cj
                    C[:, j] = (C[:, j] + np.abs(C[:, j])) / 2  # replace negative values with zeros
                    cj = cj + C[:, j]

                # merge components if their spectra are almost same
                if itr > 3:
                    SS = S.T @ S
                    i, j = np.where(SS >= self.threshold_merge)
                    m = i < j
                    i, j = i[m], j[m]
                    for n in range(len(i)):
                        S[:, j[n]] = 1 / np.sqrt(num_X_ch)
                        C[:, i[n]] = np.sum(C[:, np.r_[i[n], j[n]]], axis=1)
                        C[:, j[n]] = 0
                if np.sum(cj) < eps:
                    C[:, :] = eps

                # update lambda(ARD parameters)
                L = (np.sum(C, axis=0) + self.beta_) / (num_X_xy + self.alpha + 1) + eps
                lambdas[itr, :] = L.copy()

                # update sigma2 (the variance of additive Gaussian noise)
                X_est = C @ S.T  # reconstructed data matrix
                sigma2 = np.mean((X - X_est) ** 2)

                # object function (negative log likelihood)
                obj[itr] = num_X_xy * num_X_ch / 2 * np.log(2 * np.pi * sigma2) + num_X_xy * num_X_ch / 2  # MSE
                obj[itr] = obj[itr] + (L ** (-1)).T @ (np.sum(C, axis=0) + self.beta_).T \
                           + (num_X_xy + self.alpha + 1) * np.sum(np.log(L), axis=0) + const

                # check of convergence
                if (itr > 1) & (np.abs(obj[itr - 1] - obj[itr]) < 10 ** (-10)):
                    obj = obj[0:itr]
                    lambdas = lambdas[0:itr, :].copy()
                    break

            # choose the best result
            if obj_best[-1] > obj[-1]:
                obj_best = obj.copy()
                C_best = C.copy()
                S_best = S.copy()
                lambdas_best = lambdas.copy()

        # for learning curve of object function
        self.obj_fun_ = obj_best
        # replace tiny values with zeros
        C_best[C_best < eps] = 0
        S_best[S_best < eps] = 0
        L_best = (np.sum(C, axis=0) + self.beta_) / (num_X_xy + self.alpha + 1)
        k = np.argsort(-L_best)
        num_comp_best = np.sum(L_best[k] > eps)
        ks = k[:num_comp_best]
        self.C_, self.S_, self.L_ = C_best[:, ks], S_best[:, ks], L_best[ks]
        self.lambdas_ = lambdas_best[:, k] # leave all values to draw learning curve of ARD
        X_est = self.C_ @ self.S_.T  # reconstructed data matrix
        self.sigma2_ = np.mean((X - X_est) ** 2)

        return self


    def plot_ard(self, figsize=list()):
        '''
        Plot learning curve of component intensities (#iterations vs intensities)
        Parameters
        ----------
        figsize: the vertical and horizontal size of the figure
        '''
        if len(figsize) == 0:
            plt.figure()
        else:
            plt.figure(figsize=figsize)
        for k in range(self.n_components):
            plt.plot(self.lambdas_[:, k], label=str(k + 1))
        plt.xlabel('Iterations')
        plt.ylabel('Intensity')
        plt.xlim([0, self.lambdas_.shape[0]])
        plt.title('Intensity of components')
        plt.legend()
        plt.show()
