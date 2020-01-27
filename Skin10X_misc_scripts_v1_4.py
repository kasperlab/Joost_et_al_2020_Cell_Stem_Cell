################################################################################
################### 10X -- MISCELLANEOUS SCRIPTS ####################
################################################################################

"""
A variety of smaller scripts for data input, data wrangling and transformation,
data plotting and data analysis.
Version: Python 3
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import os, math, datetime, random, itertools
from collections import Counter
import numpy as np
import scipy.stats
import pandas as pd
import scanpy.api as sc
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform

from Skin10X_plot_v1_0 import *


################################################################################
################################# DATA INPUT ###################################
################################################################################

def create_ID():
    
    """
    Creates experiment ID (YmdHm) to identify output"
    """

    exp_id = datetime.datetime.now().strftime("%Y%m%d%H%M")
    
    print("\nThe experiment ID is %s" % (exp_id))
    
    return exp_id
    
################################################################################

def saveData_v1(dataset, path, id_, name):
    
    """
    Saves pd.DataFrame or pd.Series to csv.
    ----------
    dataset: [pd.DataFrame] or [pd.Series].
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    """
            
    dataset.to_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t')
    
################################################################################

def saveData_to_pickle_v1(dataset, path, id_, name):
    
    """
    Saves pd.DataFrame or pd.Series to pickle.
    ----------
    dataset: [pd.DataFrame] or [pd.Series].
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    """

        
    dataset.to_pickle('%s/%s_%s.txt' % (path, id_, name))

################################################################################

def saveData_to_h5_v1(df, path, id_, name):

	"""
    Saves pd.DataFrame to h5.
    ----------
    df: [pd.DataFrame]
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    """

    df.to_hdf('%s/%s_%s.h5' % (path, id_, name),
              key = name,
              mode = 'w',
              format = 'fixed')

################################################################################

def loadData_v1(path, id_, name, dform):
    
    """
    loads pd.DataFrame or pd.Series from csv.
    ----------
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    datatype: 'DataFrame' or 'Series'.
    ----------
    returns [pd.DataFrame] or [pd.Series]
    """
    
    if dform == 'DataFrame':
        return pd.read_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t', header = 0, index_col = 0, 
                           low_memory = False, squeeze = True)
    
    elif dform == 'Series':
        return pd.read_csv('%s/%s_%s.txt' % (path, id_, name), sep = '\t', header = None, index_col = 0, 
                           low_memory = False, squeeze = True)
    
################################################################################
    
def loadData_from_pickle_v1(path, id_, name):
    
    """
    loads pd.DataFrame or pd.Series from csv.
    ----------
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    ----------
    returns [pd.DataFrame] or [pd.Series]
    """
    
    return pd.read_pickle('%s/%s_%s.txt' % (path, id_, name))

################################################################################

def loadData_from_h5_v1(path, id_, name):

	"""
    loads pd.DataFrame or pd.Series from h5.
    ----------
    path: path to saving location.
    id_: experimental ID (e.g. YYMMDDHHMM).
    name: name of saved file. Format: /path/ID_name.
    ----------
    returns [pd.DataFrame]
    """
    
    return pd.read_hdf('%s/%s_%s.h5' % (path, id_, name), key = name)

################################################################################

def to_scanpy(df):

	"""
	Transforms a pd.DataFrame to a scanpy AnnData object with row names as variables and 
	column names as observations.
	----------
	df: [pd.DataFrame]
	----------
	returns scanpy AnnData object
	"""
    
    X = df
    obs = pd.DataFrame()
    obs['CellID'] = X.columns
    var = pd.DataFrame()
    var['Gene'] = X.index
    scdata = sc.AnnData(np.array(X.T), obs = obs, var = var)

    return scdata

################################################################################

def sc2pd(scdata, layer, sparse = True):

	"""
	Transforms a scanpy AnnData object into a (dense) pd.DataFrame with var_names as row and
	obs_names as column IDs.
	----------
	scdata: scanpy AnnData object.
	layer: layer of the AnnData object to be transformed.
	sparse: whether layer of the AnnData is in sparse format.
	----------
	returns pd.DataFrame
	"""
    
    if sparse:
        return pd.DataFrame(scdata.layers[layer].T.todense(),
                            index = scdata.var_names,
                            columns = scdata.obs_names)
    
    else:
        return pd.DataFrame(scdata.layers[layer].T,
                            index = scdata.var_names,
                            columns = scdata.obs_names)

################################################################################
################# DATA TRANSFORMATION AND FEATURE SELECTION ####################
################################################################################

def fuse_genes_v1(df):

	"""
	Fuses gene isoforms combining count data.
	----------
	df: pd.DataFrame
	----------
	returns pd.DataFrame
	"""

    output = df.copy()
    
    counter_ix = Counter(df.index)
    
    for ix in set([ix for ix in df.index if counter_ix[ix] > 1]):
        tmp = df.loc[[ix]].sum(axis=0)
        output = output.drop(ix)
        output.loc[ix] = tmp
        
    return output

################################################################################

def dropNull_v2X(dataset, cutoff_mean = 0):

	"""
    Drops genes with mean expression below or at a cutoff.
    ----------
    dataset: pd.Dataframe of m cells x n genes.
    cutoff_mean: mean expression cutoff. Exclusive.
    ----------
    pd.Dataframe with gene below cutoff excluded
    """
    
    print('\nDropping unexpressed genes from dataset')
    dataset = dataset.drop(dataset[dataset.mean(axis = 1) <= cutoff_mean].index)
    
    return dataset
    
################################################################################

def cellCutoff(dataset, cutoff):
    
    """
    Removes all observations / cells whose total number of transcripts lies below a
    specified cutoff.
    ----------
    dataset: seq dataset [pd.DataFrame] containing m cells x n genes.
    cutoff: number of total transcript / molecule count [int] under which a cell is dropped from the dataset.
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
    
    print('\nRemoving cells with less than %s transcripts' % (cutoff))
    dataset = dataset[dataset.columns[dataset.sum() >= cutoff]]
    
    return dataset

################################################################################

def log2Transform(dataset, add = 1):
    
    """
    Calculates the binary logarithm (log2(x + y)) for every molecule count / cell x in dataset. 
    Unless specified differently, y = 1.
    ----------
    dataset: [pd.DataFrame] containing m cells x n genes.
    add: y [float or int] in (log2(x + y)).
    ----------
    returns seq dataset [pd.DataFrame] containing m cells x n genes.
    """
    
    print('\nCalculating binary logarithm of x + %s' % (add))
    dataset = np.log2(dataset.astype(float) + float(add))
    
    return dataset

################################################################################

def select_features_log2_var_polyfit_v2(data, cutoff_mean, n_features, return_all=False):

	"""
	Selects high variance features by fitting a 2nd degree polynomial.
	----------
	data: [pd.DataFrame] containing m cells x n genes.
    cutoff_mean: mean expression cutoff. Exclusive.
    n_features: number of highest variance genes to select.
    return_all: whether to return raw data (residuals and fitting parameters).
    ----------
    returns pd.DataFrame containing [n_features] highest variance features
	"""
    
    ####################
    
    def log2_var_polyfit(dataset):
    
        """
        Helper function.
        """
        
        data_mean = dataset.mean(axis = 1)
        data_var = dataset.var(axis = 1)
        
        log2_mean = np.log2(data_mean + 1)
        log2_var = np.log2(data_var + 1)
            
        z = np.polyfit(log2_mean, log2_var, 2)
        
        log2_var_fit = z[0] * (log2_mean**2) + z[1] * log2_mean + z[2]
        log2_var_diff = log2_var - log2_var_fit
    
        return log2_var_fit, log2_var_diff, z

    ####################

    data = dropNull_v2X(data, cutoff_mean=cutoff_mean)
    
    print("\nAfter mean expression cutoff of %s, %s genes remain" % (cutoff_mean, len(data.index)))
    
    log2_var_fit, log2_var_diff, z = log2_var_polyfit(data)
    
    genes_sel = log2_var_diff.sort_values()[-n_features:].index
    
    draw_log2_var_polyfit(data, log2_var_diff, z, selected=genes_sel)
    
    print("\nAfter high variance feature selection, %s genes remain" % (len(genes_sel)))
    
    data = data.ix[genes_sel]
    
    data_log2 = np.log2(data + 1 )
    
    if return_all==True:
        return data_log2, log2_var_diff, z
    
    else:
        return data_log2
    
################################################################################

def pca_explained_var(data, dim = 50, **kwargs):

	"""
	Visualizes variance explained by each principal component.
	----------
	data: [pd.DataFrame] containing m cells x n genes.
	dim: number of PCs to plot.
	"""
    
    pca = PCA(n_components=dim, **kwargs)
    pca_fit = pca.fit(data.T)
    exp_var = pca_fit.explained_variance_
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(range(dim), exp_var)
    
    plt.figure(figsize = [15,10], facecolor = 'w')
    ax = plt.axes()
    
    ax.set_xlabel('Principal components')
    ax.set_ylabel('Explained Variance')
    
    ax.set_xlim(-0.5, dim-0.5)
    ax.set_ylim(0, np.max(exp_var) * 1.1)
    
    ax.scatter(range(dim), exp_var, c = 'dodgerblue', linewidth = 0, s = 100)

################################################################################

def dim_reduc_pca(data, dim, distance, inverse_transform = True, **kwargs):

	"""
	Performs principal component analysis, returning a pd.DataFrame of PCs per cells or the PCA-transformed input matrix as well
	as a cell x cell distance metric based on either metric.
	----------
	data: [pd.DataFrame] containing m cells x n genes.
	dim: number of PCs to consider.
	distance: scipy pdist compatible distance metric.
	inverse_transform: whether to calculate cell x cell distances based on the PCA-transformed input matrix (default: True)
	----------
	returns pd.DataFrame of m cells x [dim] PCs or pd.DataFrame containing the inversed transformed input matrix.
	return pd.DataFrame of pair-wise distances between m x m cells.
	"""
    
    pca = PCA(n_components=dim, **kwargs)
    
    if inverse_transform == True:
        data_pca = pd.DataFrame(pca.inverse_transform(pca.fit_transform(data.T)).T, 
                                index = data.index, columns = data.columns)
        
    else:
        data_pca = pd.DataFrame(pca.fit_transform(data.T).T, index = range(dim), columns = data.columns)
        
    return data_pca, pd.DataFrame(squareform(pdist(np.array(data_pca.T), distance)), index = data.columns, columns = data.columns)

################################################################################

def remove_cells(data, genes, cutoff):

	"""
	Simple function to select cells which express genes over a threshold for exclusion. A cell is selected if expression reaches
	the specified threshold for at least one of the specified genes.
	----------
	data: [pd.DataFrame] containing m cells x n genes.
	genes: list or array of gene names.
	cutoff: list or array of expression threshold for the genes in [genes]. Exclusive.
	----------
	returns list of cells to exclude.
	"""
    
    c_rem = []
    
    for g, c in zip(genes, cutoff):
        c_rem += list(data.loc[g][data.loc[g]>=c].index)
        
    return list(set(c_rem))

################################################################################
################### COMPARISON BETWEEN CLUSTERING METHODS ######################
################################################################################

def heatmap_diag(hm):
    
    import operator
    
    #get dict and sort
    
    d = {}
    
    for r in hm.index:
        for c in hm.columns:
            d[(r,c)] = hm.loc[r,c]
            
    r_sort, c_sort = [], []
            
    d = sorted(d.items(), key=operator.itemgetter(1), reverse = True)
    
    for i in d:
        if i[0][0] not in r_sort and i[0][1] not in c_sort:
            r_sort += [i[0][0]]
            c_sort += [i[0][1]]
    
    r_sort += [r for r in hm.index if r not in r_sort]
    c_sort += [c for c in hm.columns if c not in c_sort]
    
    return hm.loc[r_sort, c_sort]

################################################################################

def seurat_vs_AP(seurat, AP):
    
    #define output
    
    output = pd.DataFrame(index = return_unique(AP), columns = return_unique(seurat))
    
    for col in return_unique(seurat):
        ix_col = set(seurat[seurat==col].index)
        
        for row in return_unique(AP):
            ix_row = set(AP[AP==row].index)
            
            output.loc[row, col] = float(len(ix_row&ix_col)) / float(len(ix_col))
            
    return heatmap_diag(output)
    
################################################################################
############################## HELPER FUNCTIONS ################################
################################################################################

def chunks(l, n):
    """ 
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]
