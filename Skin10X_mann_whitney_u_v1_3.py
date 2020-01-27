################################################################################
############################ 10X -- MANN-WHITNEY U #############################
################################################################################

"""
Scripts for non-parametric statistical testing using Mann-Whitney U test (Wilcoxon rank sum test).
Version: Python3
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import random, itertools
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu as mwu

import rpy2.robjects as robj
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import Vector, DataFrame, FloatVector, IntVector, StrVector, ListVector, Matrix, BoolVector
#from rpy2.rinterface import RNULLType
stats = importr('stats')

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

def MWU_vs_average(data, groups, genes, dview, BH = True, log = True):

    """
    Performs MWU test for differential expression in cluster C compared to combined expression over all other clusters.
    ----------
    data: pd.DataFrame of m cells x n genes.
    groups: pd.Series of cluster identity in m cells.
    genes: list of selected genes.
    dview: ipyparallel dview object.
    BH: whether to perform Benjamini-Hochberg correction. Default: True
    log: whether to return -log10 transformed pvalues.
    -----------
    returns p-values of genes in [genes] for all clusters in [groups].
    """
        
    #########################
    
    def MWU_vs_average_helper(data, groups, gene):
        
        output = pd.DataFrame(index = [gene], columns = return_unique(groups))
                
        for gr in return_unique(groups):
            d1 = data.ix[groups[groups==gr].index]
            d2 = data.ix[groups[groups!=gr].index]
             
            try:
                output.ix[gene,gr] = mwu(d1, d2, alternative = 'greater')[1]
            except:
                output.ix[gene,gr] = 1.0
                
        return output
            
    #########################
    
    l = len(genes)
    
    output_tmp = dview.map_sync(MWU_vs_average_helper,
                                [data.ix[g] for g in genes], 
                                [groups] * l, 
                                genes)
        
    output = pd.concat(output_tmp, axis = 0)
        
    if BH == True:
        for col in output.columns:
            output[col] = stats.p_adjust(FloatVector(output[col]), method = 'BH')
            
    if log == True:
        output = -np.log10(output.astype(float))
    
    return output
    
################################################################################

def MWU_vs_groups(data, groups, genes, dview, BH = True, log = True):

    """
    Performs MWU test for differential expression in cluster C compared to each other cluster. Returned is the maximal p-value.
    ----------
    data: pd.DataFrame of m cells x n genes.
    groups: pd.Series of cluster identity in m cells.
    genes: list of selected genes.
    dview: ipyparallel dview object.
    BH: whether to perform Benjamini-Hochberg correction. Default: True
    log: whether to return -log10 transformed pvalues.
    -----------
    returns p-values of genes in [genes] for all clusters in [groups].
    """
        
    #########################
    
    def MWU_vs_groups_helper(data, groups, gene):
        
        output = pd.DataFrame(index = [gene], columns = return_unique(groups))
        
        for gr1 in return_unique(groups):
            d1 = data.ix[groups[groups==gr1].index]
            pvals = []
            
            for gr2 in [gr2 for gr2 in return_unique(groups) if gr2 != gr1]:
                d2 = data.ix[groups[groups==gr2].index]
                
                try:
                    pval_tmp = mwu(d1, d2, alternative = 'greater')[1]
                except:
                    pval_tmp = 1.0
                                        
                pvals.append(pval_tmp)
                    
            output.ix[gene, gr1] = np.max(pvals)
                
        return output.astype(float)
            
    #########################
    
    l = len(genes)
    
    output_tmp = dview.map_sync(MWU_vs_groups_helper,
                                [data.ix[g] for g in genes], 
                                [groups] * l, 
                                genes)
    
    output = pd.concat(output_tmp, axis = 0)
    
    if BH == True:
        for col in output.columns:
            output[col] = stats.p_adjust(FloatVector(output[col]), method = 'BH')
            
    if log == True:
        output = -np.log10(output.astype(float))
    
    return output
    
################################################################################

def MWU_select_features(data, groups, cutoff_mean):

    """
    Returns list of features from [data] based on mean expression threshold [cutoff_mean] in at least one group in [groups]
    """
    
    #get mean data per group
    
    mean = pd.DataFrame(index = data.index, columns = set(groups))
    
    for gr in set(groups):
        c_sel = groups[groups==gr].index
        
        mean.ix[data.index, gr] = data.ix[data.index, c_sel].mean(axis = 1)
        
    #select genes over cutoff
    
    genes_sel = mean.max(axis=1)[mean.max(axis=1)>cutoff_mean].index
    
    return genes_sel

################################################################################