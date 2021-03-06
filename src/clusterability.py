# -*- coding: utf-8 -*-
"""clusterability

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gyGeh0F4yDfsrbw84CptHzlgh_1EtXmq

# Spatial Histogram
"""

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats import entropy
import math

# Spatial Histogram for Clustering Tendency
def spaHist(arr, bins=5, n = 500):
    '''
        input: arr: an numpy array of input data in d dimension
               bins: the number of bins for computing Estimated Probability Mass Function along dimensions
               n: number of random instances for comparison
        return:  an numpy array of n KL divergence numbers between the EPMF of the 
                 input arr and the EPMFs of n randomly generated arrays 
    '''
    
    ans = np.zeros(n)

    # Compute the Estimated Probability Mass Function for the input array along all its dimensions
    # the second paramter number is for the number of binning on a dimension
    epmf_input = getEPMF(arr, bins)

    for i in range(n):
        print("Computing KL for {}th sample".format(i))
        aRandArr = getRandomArray(arr, arr.shape[0])
        epmf_rand = getEPMF(aRandArr, bins)

        kl_conv = computeKLConv(epmf_input, epmf_rand)
        ans[i] = kl_conv

    # Replace the 'inf' KL divergence value with the mean  
    kls_vals = np.where(np.isinf(ans), np.mean(ans[np.isfinite(ans)]), ans)

    return kls_vals

# Compute the KL divergence between two given EPMF
from scipy.stats import entropy
def computeKLConv(epmf_input, epmf_rand):
    return entropy(epmf_input, epmf_rand, base=2)

# Compute an empirical probability mass function for given array of point
def getEPMF(arr, bins=5, smoothing=False):
    '''
        input: arr: an numpy array of d dimension points
               bins: number of bins for computing the EPMF 
        return: an numpy array of EPMF values in the cells binned along arr's dimensions
    '''
    print("Computing an empirical probability mass function: getEPMF().")
    dims = arr.shape[1]

    # If smoothing is needed, initialize all counts with 1. 
    ans = np.zeros(int(math.pow(bins, dims)))
    if smoothing == True:
        ans = np.ones(int(math.pow(bins, dims)))
    
    # cut each dimension into bins with labels of bin indexes
    cats = np.zeros(arr.shape)
    for i in range(dims):
        cats[:, i] = pd.cut(arr[:, i], bins=bins, labels=range(0, bins))
    
    # Compute the index of the EPMF array using the 
    # category numbers of each point in the input array
    for i in range(arr.shape[0]):
        idx = 0
        for j in range(dims):
            pow = dims - 1 - j
            idx = idx + cats[i, j] * math.pow(bins, pow)
        ans[int(idx)] = ans[int(idx)] + 1 # update the counts at the cell indexed by idx


    return ans / sum(ans)

# Generate a list of m number of purely random points corresponding to the input array 
def getRandomArray(arr, m):
    '''
        input: arr: an input array in d dimensions
               m: the number of random points generated
        return: an array of m random points in the same dimension as the input array
    '''
    print("Generating a list of purely random points: getRandomArray().")
    # The list of minimum values and the list of maximum values
    # This two lists define the boundary of the area for randomly generating samples
    # We assume the input array has different scales along its dimensions
    mins = []
    maxs = []

    dims = arr.shape[1]

    for i in range(dims):
        mins.append(arr[:, i].min())
        maxs.append(arr[:, i].max())

    ans = np.zeros((m, dims))

    for i in range(m):
        ans[i] = np.random.uniform(mins, maxs)

    return ans

# Generate a random EPMF for testing purpose
import numpy.random as random
def randEPMF(n = 25):
    # random.seed(42)
    rnums = random.randint(1, 100, n)
       
    return rnums / sum(rnums)

"""# Hopkins Statistic

The Hopkins statistic is a sparse test for spatial randomness. Given a dataset $\mathbf{D}$ comprising $n$ points, we generate $t$ random susamples $\mathbf{R}_{i}$ of $m$ points each, where $m<<n$. These samples are drawn from the same data space as $\mathbf{D}$, generated uniformly at random along each dimension. Further, we also generate $t$ subsamples of $m$ points directly from $\mathbf{D}$, using sampling without replacement.

Let $\mathbf{D}_{i}$ denote the $i$th direct subsample. Next, we compute the minimum distance between each points $\mathbf{x}_{j}\in \mathbf{D}_{i}$ and points in $\mathbf{D}$

$$\delta_{min}(\mathbf{x}_{j})=\min_{\mathbf{x}_{i}\in D, \mathbf{x}_{i}\neq \mathbf{x}_{j}}\left\{ \left\Vert \mathbf{x}_{j}-\mathbf{x}_{i} \right\Vert \right\}$$
Likewise, we compute the minimum distance $\delta_{min}(\mathbf{y}_{j})$ between a point $\mathbf{y}_{j}\in \mathbf{R}_{i}$ and points in $\mathbf{D}$.

The Hopkins statistic (in $d$ dimensions) for the $i$th pair of samples $\mathbf{R}_{i}$ and $\mathbf{D}_{i}$ is then defined as 
$$ HS_{i}=\frac{\Sigma_{\mathbf{y}_{j}\in \mathbf{R}_{i}} (\delta_{min}(\mathbf{y}_{j}))^d}{\Sigma_{\mathbf{y}_{j}\in \mathbf{R}_{i}}(\delta_{min}(\mathbf{y}_{j}))^d + \Sigma_{\mathbf{x}_{j}\in \mathbf{D}_{i}}(\delta_{min}(\mathbf{x}_{j}))^d} $$

This statistic compares the nearest-neighbors distribution of randomly generated points to the same distribution for random subsets of points from $\mathbf{D}$. If the data is well clustered we expect $\delta_{min}(\mathbf{x}_{j})$ values to be smaller compared to the $\delta_{min}(\mathbf{y}_{j})$  values, and in this case $HS_{i}$ tends to be 1. If both nearest-neighbor distances are similar, then $HS_{i}$ takes on values close to 0.5, which indicates that the data is essentially random, and there is no apparent clustering. Finally, if $\delta_{min}(\mathbf{x}_{j})$ values are larger compared to $\delta_{min}(\mathbf{y}_{j})$ values, then $HS_{i}$ tends to 0, and it indicates the point repulsion, with no clustering. From the $t$ different values of $HS_{i}$ we may then compute the mean and variance of the statistic to determne whether $\mathbf{D}$ is clusterable or not.
"""

# Generate a subsample of m points directly from the given data set
import numpy as np
def generateDirectSample(arr, m):
    '''
        input: arr is an numpy array of data points
               m: the size of direct sample without replaclement
        return: arr[idxs]: a direct sample of size m from the input numpy array
                idxs: the set of random indexes
    '''
    # number of input data points
    n_points = arr.shape[0]
    if m > n_points:
        raise Exception("The required sample size is too large.")
    
    idxs = np.random.choice(range(0, n_points), size=m, replace=False)

    return arr[idxs], idxs

# Compute the mininum distance from every point in arrA to arrB
import numpy as np
import scipy as sp
def computeMinDistances(arrA, arrB, idxs=None):
    '''
        input: arrA a set of points in dimension d, typically shorter or equal to arrB
               arrB a set of points in dimension d, typically longer than arrA
               idxs: a set of indices in arrB which should not be included for computing minimum
        return: an array of minimum distances from each point in arrA to arrB
    '''
    dists = sp.spatial.distance.cdist(arrA, arrB)

    if idxs is not None:
        n_points = arrA.shape[0]  
        dists_ma = np.ma.array(dists, mask=False)
        for i in range(n_points):
            dists_ma[i, idxs[i]] = True

        return np.min(dists_ma, axis=1).data
    else:
        # return the minimum value of each row (the minimum distance from a point to arrB)
        return np.min(dists, axis=1)

import numpy as np
# Compute Hopkins Statistics for a set of points
def hopkins(arr, m):
    '''
        input: arr: a set of points in an numpy arrary in dimention d
               m: the size of sample for computing Hopkins Statistics
        return: Hopkins Statistics in (0, 1)
    '''
    Di, idxs = generateDirectSample(arr, m)
    Ri = getRandomArray(arr, m)

    dists_Di = computeMinDistances(Di, arr, idxs=idxs)
    dists_Ri = computeMinDistances(Ri,arr)

    dim = arr.shape[1]

    Ri_d_norm = np.sum(np.power(dists_Ri, dim))
    Di_d_norm = np.sum(np.power(dists_Di, dim))

    return Ri_d_norm /(Ri_d_norm + Di_d_norm)