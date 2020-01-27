################################################################################
######################### 10X -- AFFINITY PROPAGATION #########################
################################################################################

"""
Scripts for affinity propagation clustering based on sklearn-implementation.
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import random, itertools
from collections import Counter
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AffinityPropagation
from fastcluster import linkage
from polo import optimal_leaf_ordering

################################################################################
################################# MISC FUNCTIONS ###############################
################################################################################

def return_unique(groups, drop_zero = False):
    """
    Returns unique instances from a list (e.g. an AP cluster Series) in order 
    of appearance.
    """
    unique = []
    
    for element in groups.values:
        if element not in unique:
            unique.append(element)
            
    if drop_zero == True:
        unique.remove(0)
        
    return unique

################################################################################
################################# CLUSTERING ###################################
################################################################################

def AP_clustering_v4P(aff_mat, axis, preference, damping, **kwargs):
      
    """
    Defines clusters along either axis of the expression matrix using the affinity propagation algorithm 
    (Frey and Dueck, Science 2007). The scikit-learn implementation is used for AP clustering.
    
    -----
    
    aff_mat[pd.DataFrame]: precomputed affinity matrix of samples or genes.
    
    axis[int]: 0 (cells) or 1 (genes)
        
    preference[float]: AP preference parameter. If preference == None, the median affinity is used as preference.
        
    damping[float]: AP damping parameter. Must be between 0.5 and 1.0.
    
    * and additional function arguments are specified in 
    http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#sklearn.cluster.AffinityPropagation
    -----
    returns: pd.Series containing axis indices (cell or gene names) with associated cluster number.
    """
    
    ########################################
    
    def affinity_propagation(data, axis, preference, damping, **kwargs):
    
        """
        Helper around sklearn AffinityPropagation function.
        """

        af = AffinityPropagation(damping=damping, preference=preference, affinity='precomputed', **kwargs)

        if axis == 0:
            af.fit(data.T)
        elif axis == 1:
            af.fit(data)

        return af

    ########################################
    
    ### get and label AP output
    
    af = affinity_propagation(aff_mat, axis, preference, damping, **kwargs)  
        
    labels = pd.Series(af.labels_, index = aff_mat.index)
    
    return labels.sort_values()

################################################################################

def eucl_2d(data):

    """
    Return pd.DataFrame of pairwaise euclidean distance of pd.DataFrame (colums)
    """
    
    return pd.DataFrame(squareform(pdist(data, 'euclidean')), index = data.index, columns = data.index)

################################################################################

def smooth_clustering(dist, cluster, k, p):

    """
    Simple function to reassign cell identity of cluster-outliers based on nearest neighbors in 2d-embedding (UMAP) space.
    ----------
    dist: pd.DataFrame of pairwise m x m cell distances in 2d-space.
    cluster: pd.Series containing cluster identity for m cells.
    k: k nearest neighbors to be considered.
    p: threshold for outlier specification. A cell is considered an outlier if less than p of its k nearest neighbors in [dist] 
    space are assigned to the same cluster.
    ----------
    returns reordered pd.Series object with reassigned cluster identity.
    """
    
    #get k-nearest neighbors of all cells in 2d-embedding space
    
    kNN = {c:dist.loc[c].sort_values()[1:k+1].index for c in cluster.index}
    
    #find cells with less than p nearest neighbors of the same cluster and reassign them to cluster with most 
    #nearest neighbors
    
    cnt = 0
    cluster_new = cluster.copy()
    
    for c in cluster.index:
        cl_c = cluster[c]
        cl_NN = Counter(cluster[kNN[c]])
        if cl_NN[cl_c] <= p:
            cluster_new[c] = cl_NN.most_common()[0][0]
            cnt += 1
            
    print('%s/%s cells reassigned!' % (cnt, len(cluster)))
    return AP_groups_reorder_v2(cluster_new, return_unique(cluster))

################################################################################
############################ PARAMETER SELECTION ###############################
################################################################################

def AP_param_sel_v4P(data, aff_mat, axis, r_preference, r_damping, dview, criterion='BIC', **kwargs):
    
    """
    Calculates the cluster numbers and information criterion values (AIC or BIC) for affinity propagation clustering 
    in the specified range of preference and damping values.
    -----
    data[pd.DataFrame]: DataFrame of m samples x n genes.
    aff_mat[pd.DataFrame]: precomputed affinity matrix of samples or genes.
    axis[int]: 0 (cells) or 1 (genes)
    r_preference[list]: list specififying range of preference values to test.
    r_damping[list]: list specififying range of damping values to test.
    view: name of Ipython DirectView Instance for parallel computing.
    criterion[str]: 'AIC' or 'BIC'. Default = 'BIC'.
    -----
    returns pd.DataFrames containing IC values (IC) and number of groups (Ng) for preference / damping pairs.
    """
    
    #initialize output
    
    output_IC = pd.DataFrame(columns = r_preference, index = r_damping)
    output_N = pd.DataFrame(columns = r_preference, index = r_damping)
    
    #define preference and damping parameters
    
    pref, damp = zip(*[x for x in itertools.product(r_preference, r_damping)])
    
    l_map = len(pref)
    
    #do AP clustering in parallel
    
    ap = dview.map_sync(AP_clustering_v4P, 
                        [aff_mat] * l_map, 
                        [axis] * l_map,
                        pref, 
                        damp)
    
    #calculate IC value
    
    ic = dview.map_sync(calculateIC_v2P,
                        [data] * l_map, 
                        ap,
                        [axis] * l_map,
                        [criterion] * l_map)
    
    #update output DataFrames
    
    for P, D, A, I in zip(pref, damp, ap, ic):
        
        output_IC.ix[D, P] = I
        output_N.ix[D, P] = len(set(A))
    
    print(output_N)
    print(output_IC)
    print(AP_IC_findmin_v1(output_IC))
    
################################################################################

def calculateIC_v2P(data, groups, axis, criterion):
    
    """
    Calculates the Aikike (AIC) or Bayesian information criterion (BIC) using a formula described in
    http://en.wikipedia.org/wiki/Bayesian_information_criterion
    -----
    data[pd.DataFrame]: DataFrame of m samples x n genes.
    groups[pd.Series]: Series containing group identity (int) for each sample or gene in dataframe.
    axis: 0 for samples, 1 for genes.
    criterion: 'AIC' or 'BIC'.
    """
    
    #for parallel processing, import modules and helper functions to engine namespace

    
    # main formula: BIC = N * ln (Vc) + K * ln (N)
    # main formula: AIC = 2 * N * ln(Vc) + 2 * K
    # Vc = error variance
    # n = number of data points
    # k = number of free parameters
    try:
        if axis == 0:

            X = data

        elif axis == 1:

            X = data.T

        Y = groups

        N = len(X.columns)

        K = len(set(Y))

        #1. Compute pd.Series Kl containing cluster lengths

        Kl = pd.Series(index = set(Y))
        Kl_dict = Counter(Y)

        for cluster in set(Y):
            Kl[cluster] = Kl_dict[cluster]

        #2. Compute pd.DataFrame Vc containing variances by cluster

        Vc = pd.DataFrame(index = X.index, columns = set(Y))

        for cluster in set(Y):

            tmp_ix = Y[Y == cluster].index
            tmp_X_var = X[tmp_ix].var(axis = 1) + 0.05 #to avoid -inf values
            Vc[cluster] = tmp_X_var

        #3. Calculate the mean variance for each cluster

        Vc = Vc.mean(axis = 0)

        #4. Calculate the ln of the mean variance

        Vc = np.log(Vc)

        #5. Multiply Vc by group size Kl

        Vc = Vc * Kl

        #6. Calculate accumulative error variance

        Vc = Vc.sum()

        #7a. Calculate BIC

        BIC = Vc + K * np.log(N)

        #7b. Calculate AIC

        AIC = 2 * Vc + 2 * K

        #8. Return AIC or BIC value


        if criterion == 'BIC':

            return BIC

        if criterion == 'AIC':

            return AIC
        
    except: return None
        
################################################################################
        
def AP_IC_findmin_v1(data):
    
    from operator import itemgetter
    
    to_dict = {}
    
    for r in data.index:
        for c in data.columns:
            to_dict[c,r] = data.loc[r,c]
            
    return sorted(to_dict.items(), key=itemgetter(1))[0][0]
        
################################################################################
############################# CLUSTER PROCESSING ################################
################################################################################

def AP_invert_index(group_file, group):
    
    """
    Inverts indices of a group in file of cluster groups.
    ----------
    group_file: pd.Series with ordered cluster identity of m cells or n genes.
    group: ID (int) of group to be inverted.
    ----------
    returns group file with inverted groups.
    """
    
    ix_new = []
    
    for gr_tmp in return_unique(group_file):
        
        if gr_tmp == group:
            
            ix_tmp = list(group_file[group_file == gr_tmp].index)[::-1]
            
        else:
            
            ix_tmp = list(group_file[group_file == gr_tmp].index)
            
        ix_new += ix_tmp
        
    group_file_new = group_file[ix_new]
    
    return group_file_new

################################################################################

def AP_order_incluster(dist, clusters, method = 'single', metric = 'correlation'):
    
    ordered = []
    
    #iterate over clusters
    
    for cl in return_unique(clusters):
        
        c_sel = clusters[clusters==cl].index
        D = pdist(dist.loc[c_sel, c_sel])
        Z = linkage(D, method = method)
        optimal_Z = optimal_leaf_ordering(Z, D)
        leaves = sch.dendrogram(optimal_Z, no_plot = True)['leaves']
        ordered += list(c_sel[leaves])
        
    return clusters[ordered]
    
################################################################################
    
def AP_groups_reorder_v2(groups, order, link_to = None):
    
    """
    Reorders the groups in an sample or gene group Series either completely or partially
    -----
    groups: pd.Series of either samples (Cell ID) or gene (gene ID) linked to groups (int)
    order: list containing either complete or partial new order of groups
    link_to: defines which group position is retained when groups are reorded partially; default == None, groups are linked to
    first group in 'order'
    -----
    returns reordered group Series
    """
    
    # (1) Define new group order
    
    if set(order) == set(groups):
        order_new = order
        
    else:
        
        order_new = return_unique(groups, drop_zero = False)
        
        if link_to in order:
            link = link_to
        
        elif link_to not in order or link_to == None:
            link = order[0]
            
        order.remove(link)
        
        for group in order:
            
            order_new.remove(group)
            ins_ix = order_new.index(link) + 1
            gr_ix = order.index(group)
            order_new.insert(ins_ix + gr_ix, group)
            
    # (2) Reorder groups
    
    groups_new = pd.Series()
    
    for group in order_new:
        
        groups_new = groups_new.append(groups[groups == group])
        
    groups_new = groups_new
    
    return groups_new
    

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################