''' Evaluates the entropy using recursive copula splitting.
    Please cite as :
    Estimating Differential Entropy using Recursive Copula Splitting
        by Gil Ariel *OrcID and Yoram Louzoun.
        Entropy 2020, 22(2), 236;
        https://doi.org/10.3390/e22020236
        Python implementation by Daniel Libman.'''
from sympy import Matrix
from scipy.linalg import null_space
from math import log, floor, sqrt
import numpy as np
from itertools import product
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

D = 5
params = dict()
params['randomLevels'] = False                       # Levels change randomly. Otherwise sequentially.
params['nbins1'] = 1000                              # Max number of bins for 1D entropy.
params['pValuesForIndependence'] = 0.05              # p-Value for independence using correlations.
params['exponentForH2D'] = 0.62                      # Scaling exponent for pair-wise entropy.
params['acceptenceThresholdForIndependence'] = -0.7  # Threshold for pair-wide entropy to accept hypothesis of independence with p-Value 0.05.
params['minSamplesToProcessCopula'] = 5              # With fewer samples, assume the dimensions are independent.
params['numberOfPartitionsPerDim'] = 2               # For recursion: partition copula (along the chosen dimension) into numberOfPartitionsPerDim equal parts.


def H1Dspacings(xs, sort=False, log=np.log):
    """Calculate the entropy of 1D data using m_N spacings.
        When sort=False (defult) Assume xs is ordered.
    """
    if sort:
        xs = np.sort(xs)
    n=len(xs)
    mn = floor(n**(1/3-0.01))
    mnSpacings = xs[mn:] - xs[:-mn] #Python starts from zero
    return (np.sum(log(mnSpacings)) + log(n/mn)*(n-mn))/n


def H1Dbins(xs, params=params, support=0):
    """ Calculate the entropy of 1D data using bins.
        if support=0 no support is used.
        support tupple: (x1, x2) is the range to use.
    """
    n = len(xs)
    nbins = max(min([params['nbins1'], floor(n ** 0.4), floor(n / 10)]), 1)
    if type(support) == int and support == 0:
        p, edges = np.histogram(xs, nbins, density=True)  # its the same as pdf, we use the width of the bin c_i/(w_i*n)
    else:
        edges = np.linspace(*support, nbins + 1)
        p, _ = np.histogram(xs, bins=edges, density=True)
    mask = p > 0  # to prevent log(0)
    H = -(edges[1] - edges[0]) * np.sum(p[mask] * np.log(p[mask]))  # integral by parts
    return H

def H2D(xs, params=params, log=np.log):
    """ Calculate the entropy of 2D data compatly supported in [0,1]^2.
    """
    n = xs.shape[0]
    if n<4:
        return 0
    nbins = max(min([params['nbins1'], floor(n**0.2), floor(n/10)]), 2) #number of bins
    pys = np.floor(xs * nbins) # The partition number in each dimension.
    py1 = pys[:,0] + pys[:,1] * nbins # Enumerate the partitions from 1 to npartitions^2.
    edges = np.arange(-0.5, nbins**2-0.5+1)
    counts,_ = np.histogram(py1, bins=edges) #  1D histogram
    mask=counts>0
    H = -np.sum(counts[mask]*log(counts[mask])) /n + log(n / nbins / nbins)
    return H


def estimateEntropyUsingCopulas(xs, support=0, level=0, params=params):
    ''' Evaluates the entropy using recursive copula splitting.
        Please cite as :
        Estimating Differential Entropy using Recursive Copula Splitting
            by Gil Ariel *OrcID and Yoram Louzoun.
            Entropy 2020, 22(2), 236;
            https://doi.org/10.3390/e22020236
            Python implementation by Daniel Libman.
        Input:
            xs: Samples. Each row is an iid sample.
            support: The support of the distribution, assumed to be a rectangle.
                     A 2xD array. First row are lower limits. Second row are upper limits in each dimension.
                     0 (Default): not known the algorithm gone use 1D mnSpacing to calculate entropy for level=0.
            level: Depth in the tree. Default is 0.
        Output:
            H: Estimated entropy
    '''
    D = xs.shape[1]
    n = xs.shape[0]

    # Stoping condition no data
    if n == 0:
        return 0
    # Calculate the empirical cumulative distribution along each dimension
    #    (i.e., the integral transform of marginals).
    H1s = 0
    ys = np.zeros((n, D))
    y1 = np.zeros(n)
    for d in range(D):
        x1 = xs[:, d]
        idx = np.argsort(x1)
        x1s = x1[idx]

        # 1D - entropy of marginals
        if type(support) == int and support == 0:
            # The range is not known, use m_n spacing method.
            H1 = H1Dspacings(x1s);
        else:
            # The range is given, use binning.
            H1 = H1Dbins(x1s, params, support[:, d])
        H1s = H1s + H1
        # Rank of marginal
        y1[idx] = (np.arange(n) + 0.5) / n
        ys[:, d] = y1
    # can clear x1 xs; %No need for these any longer
    del(xs)
    del(x1)

    # we use call by ref so it's not important
    # Stop 1:
    if n < params['minSamplesToProcessCopula'] * D or D == 1:
        return H1s

    R, P = np.hsplit(np.array([pearsonr(ys[:, i], ys[:, j]) for i, j in product(range(D), range(D))]), 2)
    R = R.reshape((-1, D))
    P = P.reshape((-1, D))
    # [R,P]=corr(ys); % Calculate the Peterson correlation coefficient of all pairs.
    #                 % Since the ys are the ranks, it is equal to the Spearman correlation of the xs.
    #                 % P is the P-value matrix of the hypothesis that dimension pairs are independent.
    np.fill_diagonal(P, 0)
    isCorrelated = P < params[
        'pValuesForIndependence']  # A Boolean matrix. ==true is the corresponding pair is correlated.
    if isCorrelated.sum() < D ** 2:  # Not all pairs are correlated
        # Do more checks for pairs that are not correlated
        c = np.argwhere(isCorrelated == 0)
        nFactor = n ** params['exponentForH2D']
        for i, j in c:
            if i > j:
                # Calculate pair-wide entropy = mutual information (because marginals are U(0,1).
                H2 = H2D(ys[:, [i, j]], params)
                isCorrelated[i, j] = H2 * nFactor < params['acceptenceThresholdForIndependence']
                isCorrelated[j, i] = isCorrelated[i, j]

                # Partition isCorrelated into blocks
    # using the Laplacian kernal to find components of the graph
    nCorrelated = isCorrelated.sum(axis=0)  # cov metrix A = A^T
    L = isCorrelated - np.diag(nCorrelated)
    #np.array(symL.nullspace()).T
    Z = Matrix(L).nullspace()  # python unlike matlab doesn't give rational (Z) answer norm(v1)=1
    Z = np.hstack(Z)
    # Z = np.array(Z).T
    Z = Z != 0  # get 1, 0 values of components
    nBlocks = Z.shape[1]
    if nBlocks >= 2:
        H = H1s
        # Split into blocks
        for c in range(nBlocks):
            clusterSize = Z[:, c].sum()
            if clusterSize > 1:  # Blocks of size 1 are a marginal. The distribution is U(0,1), so its entropy is 0.
                dimsInCluster = np.argwhere(Z[:, c]).reshape(-1)
                ysmall = ys[:, dimsInCluster]  # Samples of the block
                smallD = len(dimsInCluster)  # The dimension of the block
                unitCube = np.array([[0] * smallD, [1] * smallD])  # The support is always [0,1]^smallD.
                Hpart = estimateEntropyUsingCopulas(ysmall, unitCube,
                                                    level + 1)  # Calculate the entropy of the reduced block.
                # Comment - the blocks number gone eventually reduced to one component and then the algoritem gone stop
                # Add the entropy of the block
                H += Hpart
        return H
    # Stop 2:
    if n < params['minSamplesToProcessCopula']*D:
        H = H1s
        return H

    # Find which dimensions are most correlated with others.
    R2 = R ** 2  # R^2 value for correlation. (This time working with R not P)
    np.fill_diagonal(R2, 0)
    Rsum = R2.sum(axis=0)
    # np.argwhere(np.bitwise_and(Rsum>28, Rsum<=36)).reshape(-1)
    largeDims = np.argwhere(np.bitwise_and(Rsum.max() - Rsum < 0.1,
                                           Rsum > 0)).reshape(
        -1)  # List of dimenstions which are highly correlated with others (can be 0, 2, 3, ..., but not 1).
    # Comment: Rsum>0 - Rsum can be equal to zero for example [[1,0],[0,1]]
    if len(largeDims) == 0:
        # Correlations are small, yet variables are dependent. All dims are equally problematic.
        largeDims = np.arange(D)
    nLargeDims = len(largeDims)
    #  Pick one of the dims in largeDims
    if params['randomLevels']:
        # Choose randomly.
        maxCorrsDim = largeDims[np.random.randint(nLargeDims)]
    else:
        # Choose sequentially.
        maxCorrsDim = largeDims[level % nLargeDims]

    unitCube = np.array([[0] * D, [1] * D])
    Hparts = np.zeros(params['numberOfPartitionsPerDim'])
    # split the data along Dim maxCorrsDim into params.numberOfPartitionsPerDim equal parts.
    for prt in range(params['numberOfPartitionsPerDim']):
        # Range of data is [f,l)
        f = prt / params['numberOfPartitionsPerDim']
        # Closing the last part
        if prt + 1 == params['numberOfPartitionsPerDim']:
            l = 2
        else:
            l = (prt + 1) / params['numberOfPartitionsPerDim']

        mask = np.bitwise_and(ys[:, maxCorrsDim] >= f, ys[:, maxCorrsDim] < l)  # True/False value for relevant row
        # Subset of data.
        y1 = ys[mask]
        # Scale back to [0,1].
        y1[:, maxCorrsDim] = (y1[:, maxCorrsDim] - f) * params['numberOfPartitionsPerDim']
        # Entropy of subset.
        Hparts[prt] = estimateEntropyUsingCopulas(y1, unitCube, level + 1)

    # Add the entropies of all subsets.
    H = H1s + Hparts.sum() / params['numberOfPartitionsPerDim']
    return H


if __name__ == '__main__':
    D = 5
    nsamples = 10000
    # example 1 - uniform distribution in [0,1]^D
    # %generate nsamples independent samples of i.i.d. random variables,
    # %       uniformally distributed in [0,1]^D
    print("example 1 - uniform distribution in [0,1]^%s" % (D))
    print("generate %s samples independent samples of i.i.d. random variables uniformally distributed in [0,1]^%s" % (
    nsamples, D))
    x = np.random.rand(nsamples, D)
    # define the support of the distribution
    support = np.array([[0] * D, [1] * D])
    H1 = estimateEntropyUsingCopulas(x, support)
    print("Entropy %s" % (H1))
    print("***********************************************************************")
    # % example 2 - normal distribution in \R^D
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # %generate nsamples independent samples of i.i.d. random variables,
    # %       with a standard normal distribution in \R^D
    print("example 2 - normal distribution in R^%s" % (D))
    print("generate %s samples independent samples of i.i.d. with a standard normal distribution in  R^%s" % (
    nsamples, D))
    y = np.random.randn(nsamples, D)
    # define the support of the distribution
    H2 = estimateEntropyUsingCopulas(y)
    print("Entropy %s" % (H2))
    print("***********************************************************************")

    D = 5
    # example 3 - distribution in [0,1]^D with higher density in [0,0.5]
    # %generate nsamples independent samples of i.i.d. random variables,
    z1 = np.random.rand(nsamples, D) * 0.5
    z2 = np.random.rand(nsamples, D) * 0.5 + 0.5
    z1_prob = 0.80
    mask = np.random.rand(nsamples, D) < z1_prob
    z = z1 * mask + z2 * (1 - mask)
    print("example 3 - distribution in [0,1]^%s with higher density in [0,0.5]]^%s" % (D, D))
    print("generate %s samples independent samples of i.i.d. random variables")
    # x = np.random.rand(nsamples,D)
    # define the support of the distribution
    support = np.array([[0] * D, [1] * D])
    H3 = estimateEntropyUsingCopulas(z, support)
    print("Entropy %s" % (H3))
    print("***********************************************************************")

    D = 5
    #example 4 - normal distribution in \R^D
    # %generate nsamples independent samples of i.i.d. normal random variables with low variance,
    print("example 4 - normal distribution in R^%s"%(D))
    print("generate %s samples independent samples of i.i.d. with a low variance normal distribution in  R^%s"%(nsamples,D))
    m = np.random.randn(nsamples,D) * sqrt(0.2)
    H4=estimateEntropyUsingCopulas(m)
    print("Entropy %s"%(H4))
    print("***********************************************************************")

    D = 5
    # example 5 - normal distribution in \R^D
    # %generate nsamples independent samples of i.i.d. normal random variables with high variance,
    print("example 5 - normal distribution in R^%s" % (D))
    print("generate %s samples independent samples of i.i.d. with a high variance normal distribution in  R^%s" % (
    nsamples, D))
    u = np.random.randn(nsamples, D) * sqrt(100)
    H5 = estimateEntropyUsingCopulas(u)
    print("Entropy %s" % (H5))
