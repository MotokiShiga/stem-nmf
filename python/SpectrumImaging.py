import numpy as np
import numpy.linalg as lin
from scipy.special import gammaln
from numba import *
import matplotlib.pyplot as plt

###########################################################################
#
# NMF (Nonnegative Matrix Factorization) for Spectrum Imaging Data Analysis
#
#                       (c) Motoki Shiga, Gifu University, Japan
#
#  Reference
#  [1] Motoki Shiga, Kazuyoshi Tatsumi, Shunsuke Muto, Koji Tsuda,
#      Yuta Yamamoto, Toshiyuki Mori, Takayoshi Tanji,
#      "Sparse Modeling of EELS and EDX Spectral Imaging Data
#        by Nonnegative Matrix Factorization",
#      Ultramicroscopy, Vol.170, p.43-59, 2016.
###########################################################################

# NMF with soft orthogonal constraint
def nmf_so(X, K, wo, reps=3, itr_max=100):
    #
    # NMF with soft orthogonal constraint
    #
    #
    # -- INPUT --------------------------------------------------------------
    #
    #   X      : matrix with the size of (num_xy x num_ch)
    #            num_xy: the number of measurement points on specimen
    #            num_ch: the number of spectrum channels
    #   K      : the number of components
    #   wo     : weight for the orthologal constraint on G
    #   reps   : the number of initializations
    #   itrMax : the maximum number of updates
    #
    #
    # -- OUTPUT -------------------------------------------------------------
    #
    #   C_best   : densities of components at each point
    #   S_best   : spectrums of components
    #   objs_best : learning curve (error value after each update)
    #

    num_xy, num_ch = X.shape
    obj_best = np.inf
    print('Running NMF with Soft Orthogonal constraint....')
    for rep in range(reps):
        print(str(rep+1)+'th iteration of NMF-SO algorithm')

        ### initialization
        obj = np.zeros(itr_max)
        C = np.ones((num_xy,K))
        for j in range(K):
            C[:,j] = C[:,j] / (np.sqrt(C[:,j].T@C[:,j]) + 1e-16)
        cj = np.sum(C,axis=1)
        i = np.random.choice(num_xy,K)
        S = X[i,:].T

        ### main loop
        for itr in range(itr_max):
            #update S
            XC = X.T @ C
            C2 = C.T @ C
            for j in range(K):
                S[:,j] = XC[:,j] - S@C2[:,j] + C2[j,j]*S[:,j]
                S[:,j] = (S[:,j] + np.abs(S[:,j]))/2 #replace negative values with zeros

            #update C
            XS = X @ S
            S2 = S.T @ S
            for j in range(K):
                cj     = cj - C[:,j]
                C[:,j] = XS[:,j] - C@S2[:,j] + S2[j,j]*C[:,j]
                C[:,j] = C[:,j] - wo*(cj.T@C[:,j])/(cj.T@cj)*cj
                C[:,j] = (C[:,j] + np.abs(C[:,j]))/2 #replace negative values with zeros
                C[:,j] = C[:,j] / (np.sqrt(C[:,j].T@C[:,j])) #normalize
                cj     = cj + C[:,j]

            #cost function
            X_est = C @ S.T  #reconstracted data matrix
            obj[itr] = lin.norm(X - X_est, ord='fro' )/X.size

            #check of convergense
            if(itr>1)&(np.abs(obj[itr-1] - obj[itr]) < 10**(-10)):
                obj = obj[0:itr]
                print('# updates: ' + str(itr))
                break

        #choose the best result
        if(obj_best>obj[-1]):
            objs_best = obj.copy()
            C_best = C.copy()
            S_best = S.copy()

    return(C_best,S_best,objs_best)


# NMF with ARD(Automatic Relevance Determination) prior
def nmf_ard_so(X, K, wo, a=1+10**(-15), reps=3,itr_max=100, theta_thre = 0.99):
    #
    # NMF with ARD(Automatic Relevance Determination) prior
    #
    #
    # -- INPUT --------------------------------------------------------------
    #
    #   X          : matrix with the size of (num_xy x num_ch)
    #                num_xy: the number of measurement points on specimen
    #                num_ch: the number of spectrum channels
    #   K          : the number of components
    #   wo         : weight for the orthologal constraint on G
    #   a          : hyper-parameter of the ARD prior (The smaller the value ofa, the sparser C)
    #   reps       : the number of initializations
    #   itrMax     : the maximum number of updates
    #   theta_thre : the threshold if two components should be merge
    #
    #
    # -- OUTPUT -------------------------------------------------------------
    #
    #   C_best       : densities of components at each point
    #   S_best       : spectrums of components
    #   L_best       : the final values of lambda
    #   lambdas_best : the log of updated lambda
    #   objs_best    : learning curve (error value after each update)
    #


    eps = np.finfo(np.float64).eps
    num_xy, num_ch = X.shape
    mu_X = np.mean(X)
    b = mu_X*(a-1)*np.sqrt(num_ch)/K
    const = K*(gammaln(a) - a*np.log(b))
    sigma2 = np.mean( X**2 )
    obj_best = np.inf
    print('Running NMF with ARD and Soft Orthogonal constraint....')
    for rep in range(reps):
        print(str(rep) + 'th iteration of NMF-ARD-SO algorithm')

        #--- Initialization ------
        C = (np.random.rand(num_xy,K) + 1)*(np.sqrt(mu_X/K))
        L = (np.sum(C,axis=0) + b)/(num_ch+a+1)
        cj = np.sum(C,axis=1)
        i = np.random.choice(num_xy,K)
        S = X[i,:].T
        for j in range(K):
            c = (np.sqrt(S[:,j].T@S[:,j])) #normalize
            if(c>0):
                S[:,j] = S[:,j] / c
            else:
                S[:,j] = 1/np.sqrt(num_ch)
        obj = np.zeros(itr_max)
        lambdas = np.zeros((itr_max,K))
        #-------------------------

        for itr in range(itr_max):
            #update S
            XC = X.T @ C
            C2 = C.T @ C
            for j in range(K):
                S[:,j] = XC[:,j] - S@C2[:,j] + C2[j,j]*S[:,j]
                #replace negative values with zeros (tiny positive velues)
                S[:,j] = (S[:,j] + np.abs(S[:,j]))/2
                c = (np.sqrt(S[:,j].T@S[:,j])) #normalize
                if(c>0):
                    S[:,j] = S[:,j] / c
                else:
                    S[:,j] = 1/np.sqrt(num_ch)

            #update C
            XS = X @ S
            S2 = S.T @ S
            for j in range(K):
                cj     = cj - C[:,j]
                C[:,j] = XS[:,j] - C@S2[:,j] + S2[j,j]*C[:,j]
                C[:,j] = C[:,j] - sigma2/L[j]
                if(wo>0):
                    C[:,j] = C[:,j] - wo*(cj.T@C[:,j])/(cj.T@cj)*cj
                C[:,j] = (C[:,j] + np.abs(C[:,j]))/2 #replace negative values with zeros
                cj     = cj + C[:,j]

            # merge procedure
            if(itr>3):
                SS = S.T @ S
                i, j = np.where(SS >= theta_thre)
                m = i < j
                i, j = i[m], j[m]
                for n in range(len(i)):
                    S[:, j[n]] = 1 / np.sqrt(num_ch)
                    C[:, i[n]] = np.sum(C[:, np.r_[i[n], j[n]]], axis=1)
                    C[:, j[n]] = 0
            if(np.sum(cj)<eps):
                C[:,:] = eps

            #update lambda(ARD parameters)
            L = (np.sum(C,axis=0) + b)/(num_xy+a+1) + eps
            lambdas[itr,:] = L.copy()

            #update sigma2
            X_est = C @ S.T  #reconstracted data matrix
            sigma2 = np.mean((X-X_est)**2)

            #cost
            obj[itr] = num_xy*num_ch/2*np.log(2*np.pi*sigma2) + num_xy*num_ch/2  #MSE
            obj[itr] = obj[itr] + (L**(-1)).T @ (np.sum(C,axis=0)+b).T \
                       + (num_xy+a+1)*np.sum(np.log(L),axis=0) + const

            #check of convergense
            if(itr>1)&(np.abs(obj[itr-1] - obj[itr]) < 10**(-10)):
                obj = obj[0:itr]
                lambdas = lambdas[0:itr,:].copy()
                #print('# updates: ' + str(itr))
                break

        #choose the best result
        if(obj_best>obj[-1]):
            objs_best = obj.copy()
            C_best = C.copy()
            S_best = S.copy()
            # L_best = L.copy()
            lambdas_best = lambdas.copy()

    #replace tiny values with zeros
    C_best[C_best<eps] = 0
    S_best[S_best<eps] = 0
    L_best = (np.sum(C,axis=0) + b)/(num_xy+a+1)
    k = np.argsort(-L_best)
    C_best, S_best, L_best = C_best[:,k], S_best[:,k], L_best[k]
    lambdas_best = lambdas_best[:,k]

    return(C_best, S_best, L_best, lambdas_best, objs_best)


