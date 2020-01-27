################################################################################
##################### SKIN10X -- PSEUDOTEMPORAL ORDERING #######################
################################################################################

"""
Scripts for the pseudotemporal ordering of cells based on the PQ-Tree approach
introduced by Magwene et al. and Trapnell et al.
Version: Python 3
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import itertools
from scipy.spatial.distance import pdist, squareform

from Skin10X_misc_scripts_v1_4 import *

from rpy2 import robjects
from rpy2.robjects import FloatVector
from rpy2.robjects.packages import importr

rbase = importr('base')
rvgam = importr('VGAM', robject_translations = {"nvar.vlm": "nvar__vlm"})
stats = importr('stats')

################################################################################
################################ MAIN FUNCTIONS ################################
################################################################################

def PTO_create_MST(dist_mat, prog = 'neato'):
        
    """
    Creates a MST (Prim's algorithm) based on the euclidean distance of a sample x sample distance matrix.
    -----
    dataset: distance matrix as pd.DataFrame of m samples x m samples
    prog: graphviz prog; default = 'neato'
    -----
    returns MST as networkx Graph and spatial positions from graphviz layout as dictionairy
    """
    
    G = nx.Graph()
    
    G.add_nodes_from([node for node in dist_mat.index])

    for node in G.nodes():
        
        tmp_edges = [(node, index, {'weight': dist_mat[node].loc[index]}) for index in dist_mat[node].index]
        G.add_edges_from(tmp_edges)
    
    MST = nx.minimum_spanning_tree(G)
    
    MST_pos = nx.drawing.nx_agraph.pygraphviz_layout(MST, prog=prog)
    
    return MST, MST_pos

################################################################################

def PTO_diameter_path(MST, return_edges = False):
    
    """
    Finds the diameter path of an MST using Dijkstras algorithm.
    ---
    MST: minimum spanning tree as networkx Graph
    ---
    returns diameter path as list and diameter edges as networkx Graph
    """
    
    if len(MST.nodes()) == 1:
        return [node for node in MST.nodes()]
    
    #1. Calculate node degree and find terminal nodes (degree == 1):
    
    node_degr = MST.degree()
    node_term = [node[0] for node in node_degr if node[1] == 1]
    
    #2. Find shortest paths between all nodes (Dijkstra):
    
    dijkstra_all = dict(nx.all_pairs_dijkstra_path_length(MST, weight = 1))
    
    #3. Find longest path between all combinations of terminal nodes
    
    path_max = ''
    len_path_max = 0
    
    for node_pair in itertools.combinations(node_term, 2): 
        len_path_tmp = dijkstra_all[node_pair[0]][node_pair[1]]
    
        if len_path_tmp > len_path_max: 
            len_path_max = len_path_tmp
            path_max = node_pair
    
    #4. Return path
    
    diam_path = nx.dijkstra_path(MST, path_max[0], path_max[1])
    diam_edges = nx.Graph()
    diam_edges.add_nodes_from(diam_path)
    diam_edges.add_edges_from([(diam_path[pos],diam_path[pos+1]) for pos in range(len(diam_path) - 1)])
    
    if return_edges == True:
        
        print('Diameter path between %s and %s with lenght %s' % (path_max[0], path_max[1], len_path_max))
        return diam_edges
    
    if return_edges == False:       
        return diam_path

################################################################################

def PTO_create_pseudotemporal_ordering_v1(dist, groups, MST = None, MST_pos = None,
                                          diam_edges = None, cmap = plt.cm.jet, 
                                          return_min = 50, return_path = False):
    
    """
    Creates pseudotemporal ordering of m cells.
    ----------
    dist: pd.DataFrame with pairwise distances of m x m cells.
    groups: pd.Series with group labels of m cells (used for visualization)
    MST: nx.Graph object of minimum spanning tree. If 'None', a new MST is created.
    MST_pos: nx object containing the positions of graph embedding. If 'None', a new object is created.
    diam_edges: edges of the diameter path. If 'None', new object is created.
    """
    
    print('\nCreating MST\n')
    
    if MST == None: MST = PTO_create_MST(dist)[0]
        
    if MST_pos == None: MST_pos = PTO_create_MST(dist)[1]
    
    if diam_edges == None: diam_edges = PTO_diameter_path(MST, return_edges = True)
    
    PTO_draw_MST_groups(MST, MST_pos, diam_edges, groups, cmap = cmap)
    
    MST_ = MST.copy() #Full cell labels
    
    MST, node_dict = PTO_relabel_nodes(MST)
    
    dist_int = PTO_distance_matrix(dist, node_dict)
    
    print('\nCreating PQ-Tree\n')
    
    PQ = PTO_PQ_tree(MST)
    
    print('Finding permutations\n')
    
    PTO = PTO_PQ_simple_permutations(PQ, dist_int, return_min = return_min)
        
    PTO_coords = PTO_ordered_coordinates(PTO, node_dict, dist_int)
    
    PTO_path = PTO_ordered_path(PTO, node_dict)
    
    PTO_draw_MST_groups(MST_, MST_pos, PTO_path, groups, cmap = cmap)
    
    print('\nReturning coordinates\n')

    if return_path == True:
        
        return PTO_coords, PTO_path
    
    else:
        
        return PTO_coords
    
################################################################################
################################ HELPER FUNCTIONS ##############################
################################################################################

def PTO_pairwise_distance_eucl(dataset):

    """
    Calculates the pairwise Euclidean distance for columns in [dataset].
    """
    
    dist_mat = pd.DataFrame(squareform(pdist(dataset, 'euclidean')), index = dataset.index, columns = dataset.index) 
    return dist_mat

################################################################################

def PTO_find_path(G, source, sink):

    """
    Finds shortest path between source and sink.
    ----------
    G: nx.Graph
    source: source node
    sink: sink node
    """
    
    path = nx.dijkstra_path(G, source, sink)
    edges = []
    dist = pd.Series(0.0, index = path)
    
    d = 0.0
    
    for pos, n in enumerate(path):
        
        dist[path[pos]] = d
        try:
            e = (path[pos], path[pos+1])
            edges += [e]
            d += float(G.edges[e]['weight'])
        except: continue
            
    return path, edges, dist

################################################################################

def PTO_relabel_nodes(MST):
        
    node_dict = {}
    ix = 0

    for node in MST.nodes():
        node_dict[node] = ix
        ix += 1
    
    MST = nx.relabel_nodes(MST, node_dict)
    
    return MST, node_dict

################################################################################

def PTO_distance_matrix(dist_mat, node_dict):
    
    dist_int = dist_mat
    
    dist_int.index = [node_dict[ix] for ix in dist_mat.index]
    dist_int.columns = [node_dict[ix] for ix in dist_mat.columns]
    
    return dist_int

################################################################################

def PTO_PQ_tree(MST):
    
    diam_path = PTO_diameter_path(MST)
    
    backbone = PTO_indecisive_backbone(MST, diam_path)
    
    if backbone is None:
        return Qnode(diam_path)
    
    branches = PTO_diam_path_branches(MST, backbone)

    pqtree = []
    
    for root in backbone:
        pqtree.append(Pnode([Qnode((root,))] + \
                 Pnode([PTO_PQ_tree(s) for s in branches.subgraphs[root]]) ))

    return Qnode(pqtree)

################################################################################

def PTO_indecisive_backbone(MST, diam_path):
    
    """
    Finds indecisive backbone (between first and last node with degree >= 3 in diameter path)
    -----
    MST: minimum spanning tree as networkx Graph
    diam_path: diameter path as list of nodes
    -----
    returns indecisive backbone as list of nodes
    """
    
    #1. Calculate node degrees in MST
    
    MST_degr = MST.degree()
        
    #2. Find index of first indecisive node
    
    ix1 = None
    for ix, n in enumerate(diam_path): 
        if MST_degr[n] >= 3:
            ix1 = ix
            break
            
    if ix1 is None:
        return None
            
    #3. Find index of last indecisive node
    
    for ix, n in enumerate(diam_path[::-1]):
        if MST_degr[n] >= 3:
            ix2 = len(diam_path) - ix
            break
    
    #4. Return indecisive backbone
    
    return diam_path[ix1 : ix2+1]

################################################################################

def PTO_diam_path_branches(MST, diam_path):
    
    #1. Find branches
    
    MST_copy = MST.copy()
    branches = {}
    for node in diam_path:
        branches[node] = []
        for subnode in MST.neighbors(node):
            if subnode not in diam_path:
                MST_copy.remove_edge(node, subnode)
                branches[node].append(nx.single_source_dijkstra_path(MST_copy, subnode).keys())
                
    #2. Find subgraphs
                
    subgraphs = {}
    for root in branches.keys():
        subgraphs[root] = []
        for branch in branches[root]:
            subg_tmp = MST.subgraph(branch)
            subgraphs[root].append(subg_tmp)
            
    #3. Return results
    
    class Results:
        def __init__(self, branches, subgraphs):
            self.branches, self.subgraphs = branches, subgraphs
            
    return Results(branches, subgraphs)
    

################################################################################

def PTO_PQ_simple_permutations(pq, dist_int, return_min = 100):
    
    perms = PTO_PQ_simple_permutation_node(pq[0], dist_int, return_min = 100)
    
    for node in range(1, len(pq)):
        
        print(node)
                
        perms_growing = pd.Series()
        
        if len(pq[node]) > 7:
            
            perms_new = PTO_PQ_simple_permutation_node(pq[node], dist_int, return_min = return_min, permutation = False)
        
        else:
            
            perms_new = PTO_PQ_simple_permutation_node(pq[node], dist_int, return_min = return_min, permutation = True)
        
        for i in itertools.product([eval(ix) for ix in perms.index], [eval(ix) for ix in perms_new.index]):
    
            perms_growing[str(i[0] + i[1])] = perms[str(i[0])] + perms_new[str(i[1])] + dist_int.ix[i[0][-1], i[1][0]]
            
        perms = perms_growing.sort_values()[:return_min]

    return eval(perms.index[0])

################################################################################

def PTO_PQ_simple_permutation_node(pq_sub, dist_int, return_min = 6, permutation = True):
    
    products_inp = 'itertools.product('

    for n in pq_sub:
        if type(n) == Qnode and len(n) > 1:
            inp_tmp = [list(PTO_flatten_list(n)), list(PTO_flatten_list(n[::-1]))] 
        else:
            inp_tmp = [list(PTO_flatten_list(n))]
        products_inp += '%s, ' % (inp_tmp)
    
    products_inp = products_inp[:-2] + ')'
    
    perm = pd.Series()
    
    for product in eval(products_inp):
        
        if permutation == True:
            
            for i in itertools.permutations(product):
                i = list(PTO_flatten_list(i))
                perm[str(i)] = PTO_calculate_distance(i, dist_int)
                
        elif permutation == False:
            
            i = list(PTO_flatten_list(product))
            perm[str(i)] = PTO_calculate_distance(i, dist_int)
            i = list(PTO_flatten_list(product))[::-1]
            perm[str(i)] = PTO_calculate_distance(i, dist_int)

    return perm.sort_values()[:return_min]

################################################################################

class Qnode(list):
    def __repr__(self):
        return str(tuple(self))

class Pnode(list):
    pass

################################################################################

def PTO_flatten_list(l):
    
    for i in l:
        if isinstance(i, list) or isinstance(i, tuple):
            for j in PTO_flatten_list(i):
                yield j
        else:
            yield i

################################################################################

def PTO_ordered_path(PTO, node_dict):
    
    node_dict_inv = {v:k for k, v in node_dict.items()}
    
    G = nx.Graph()
    
    G.add_nodes_from([node_dict_inv[node] for node in PTO])
    
    G.add_edges_from([(node_dict_inv[PTO[pos]], node_dict_inv[PTO[pos+1]]) for pos in range(len(PTO) - 1)])
    
    return G

################################################################################

def PTO_ordered_coordinates(PTO, node_dict, dist_int, reverse = False):
    
    if reverse == True:
        
        PTO = PTO[::-1]
        
    node_dict_inv = {v:k for k, v in node_dict.items()}
        
    coord = pd.Series()
        
    for pos in range(len(PTO)):
        
        if pos == 0:
            
            dist = 0
            
        else:
            
            dist += dist_int.ix[PTO[pos -1], PTO[pos]]
            
        coord[node_dict_inv[PTO[pos]]] = dist
        
    return coord

################################################################################

def PTO_calculate_distance(permutation, dist_int):
    
    dist = 0
    
    for pos in range(len(permutation) - 1):
        dist += dist_int.ix[permutation[pos], permutation[pos + 1]]
        
    return dist  

################################################################################
################################### MODELING ###################################
################################################################################

def fit_vgam_v3r(data, coords, branches, root, genes, path, exp_id, name,
                df=3, neg_log = False, steps = 101, checkpoint_after = 100):
    
    from rpy2 import robjects
    from rpy2.robjects import FloatVector
    from rpy2.robjects.packages import importr

    rbase = importr('base')
    rvgam = importr('VGAM', robject_translations = {"nvar.vlm": "nvar__vlm"})
    rstats = importr('stats')
    robjects.r['options'](warn=-1)
    
    def fit_vgam_v3r_helper(data, coords, branches, root, g, df, steps, perc = 99.5): 
    
            data = data.loc[g, coords.index]
    
            #clip outliers (99.5th percentile) for each branch (only considering branch unique cells)
        
            for b in [b for b in branches.columns if b != root]:
                c_sel = branches[b][branches[b]==1][branches.sum(axis=1)==1].index
                thr = np.percentile(data[c_sel], perc)
                data[c_sel] = np.clip(data[c_sel], 0, thr).values

            data = np.rint(data)
                                      
            X = robjects.FloatVector(np.array(coords.values))
            Y = robjects.FloatVector(np.array(data))

            predict_x = np.linspace(0, np.max(coords), steps)
            X_pred = robjects.FloatVector(predict_x)

            #define output - fit

            ix1 = ['all','null'] + [b for b in branches.columns if b != root]; ix2 = [g] * len (ix1)
            out_fit = pd.DataFrame(index = list([ix1, ix2]), 
                                   columns = predict_x)
            
            #define output - lr test

            ix1 = [b for b in branches.columns if b != root]; ix2 = [g] * len (ix1)
            out_lr = pd.DataFrame(index = list([ix1, ix2]),
                                  columns = ['Chisq_null','Pr(>Chisq)_null','Pr(>Chisq)-BH_null','Chisq_all','Pr(>Chisq)_all','Pr(>Chisq)-BH_all'])
            
            try:
                
                #fit reduced model (no branch specificity)
                
                DF = robjects.DataFrame({'X':X,'Y':Y})
                
                fmla_red = robjects.Formula("Y ~ sm.ns(X, df = %s)" % df)                         
                fit_red = rvgam.vglm(fmla_red, rvgam.negbinomial, data = DF, 
                                          na_action = robjects.r['na.fail'], smart = True)
                                         
                #predict reduced model 
                                         
                DF_pred = robjects.DataFrame({'X': X_pred})
                predicted = rvgam.predict(fit_red, newdata = DF_pred, type = 'response')
                out_fit.loc['all',g] = list(predicted)
                
                #fit null model (no pseudotime dependency compared to root cells)
                
                B = robjects.FloatVector(np.array(branches[root]))
                DF = robjects.DataFrame({'Y': Y, 'B':B})

                fmla_null = robjects.Formula("Y ~ B")                         
                fit_null = rvgam.vglm(fmla_null, rvgam.negbinomial, data = DF, 
                                              na_action = robjects.r['na.fail'], smart = True)

                #predict null model (no pseudotime dependency compared to root cells)
                    
                B_pred = robjects.FloatVector([1] * len(predict_x))
                DF_pred = robjects.DataFrame({'X':X_pred, 'B':B_pred})
                predicted = rvgam.predict(fit_null, newdata = DF_pred, type = 'response')
                out_fit.loc['null',g] = list(predicted) 

                for b in [b for b in branches.columns if b != root]:
                    
                #fit branch specific (full) models
                    
                    B = robjects.FloatVector(np.array(branches[b]))
                    DF = robjects.DataFrame({'X': X, 'Y': Y, 'B': B})

                    fmla_full = robjects.Formula("Y ~ sm.ns(X, df = %s) * B" % df)
                    fit_full = rvgam.vglm(fmla_full, rvgam.negbinomial, data = DF, 
                                          na_action = robjects.r['na.fail'], smart = True)

                #predict branch specific (full) models

                    DF_pred = robjects.DataFrame({'X':X_pred, 'B':B_pred})
                    predicted = rvgam.predict(fit_full, newdata = DF_pred, type = 'response')
                    out_fit.loc[b,g] = list(predicted)

                #lr test vs. branch specific null model (no pseudotime-dependence))

                    lr_stats_null = rvgam.lrtest(fit_full, fit_null)
                    out_lr.loc[b,g]['Chisq_null','Pr(>Chisq)_null'] = lr_stats_null.do_slot('Body')[3][1], lr_stats_null.do_slot('Body')[4][1]

                #lr test vs. reduced model (no branch specificity)

                    lr_stats_red = rvgam.lrtest(fit_full, fit_red)
                    out_lr.loc[b,g]['Chisq_all','Pr(>Chisq)_all'] = lr_stats_red.do_slot('Body')[3][1], lr_stats_red.do_slot('Body')[4][1]

                return out_fit, out_lr

            except:

                return out_fit, out_lr
    
    #try to load already generated output
    
    try:
        fitted = loadData_from_pickle_v1(path, exp_id, '%s_fitted' % name)
        stats = loadData_from_pickle_v1(path, exp_id, '%s_stats' % name)
        genes_sel = [g for g in genes if g not in set(fitted.index.levels[1])]
        print('%s/%s' % (len(set(fitted.index.levels[1])), len(genes)))
                
    except:       
        fitted = pd.DataFrame()
        stats = pd.DataFrame()
        genes_sel = genes
        print('%s/%s' % (0, len(genes)))
    
    #run
    
    for i, g in enumerate(genes_sel):
        
        fitted_tmp, stats_tmp = fit_vgam_v3r_helper(data, coords, branches, root, g, df, steps)
        fitted = pd.concat([fitted, fitted_tmp], axis = 0)
        stats = pd.concat([stats, stats_tmp], axis = 0)
        
        if i != 0 and i % checkpoint_after == 0:
            print('%s/%s' % (len(set(fitted.index.levels[1])), len(genes)))
            saveData_to_pickle_v1(fitted, path, exp_id, '%s_fitted' % name)
            saveData_to_pickle_v1(stats, path, exp_id, '%s_stats' % name)
    
    #multiple testing correction with BH (per branch)
        
    for b in [b for b in branches.columns if b != root]:
        stats.loc[b,'Pr(>Chisq)-BH_null'] = rstats.p_adjust(FloatVector(stats.loc[b,'Pr(>Chisq)_null']), method = 'BH')
        stats.loc[b,'Pr(>Chisq)-BH_all'] = rstats.p_adjust(FloatVector(stats.loc[b,'Pr(>Chisq)_all']), method = 'BH')

    if neg_log == True:
        stats['Pr(>Chisq)_null'] = -np.log10(stats['Pr(>Chisq)_null'].astype(float))
        stats['Pr(>Chisq)_all'] = -np.log10(stats['Pr(>Chisq)_all'].astype(float))
        stats['Pr(>Chisq)-BH_null'] = -np.log10(stats['Pr(>Chisq)-BH_null'].astype(float))
        stats['Pr(>Chisq)-BH_all'] = -np.log10(stats['Pr(>Chisq)-BH_all'].astype(float))
        
    saveData_to_pickle_v1(fitted, path, exp_id, '%s_fitted' % name)
    saveData_to_pickle_v1(stats, path, exp_id, '%s_stats' % name)
        
    return fitted, stats

################################################################################
#################################### DRAWING ###################################
################################################################################

def PTO_draw_MST_groups(MST, MST_pos, diam_edges, groups, cmap = plt.cm.jet, node_size = 100, 
                        linewidths = 1.0, width = 0.5, width_diam = 5.0, edge_color = 'grey', 
                        edges_off = False):
    
    """
    Draws MST with group specific colormap and highlighted diameter path.
    -----
    MST: MST networkx Graph
    MST_pos: networkx position dict for MST
    diamedges: networkx Graph containing nodes and edges of MST diameter path
    sample_groups: pd.Series containing sample groups from AP clustering
    """
        
    if type(cmap) != dict:
        cm = cmap
        cmap = {}
        for ix, gr in enumerate(return_unique(groups)):
            cmap[gr] = cm(float(ix) / len(set(groups)))
    
    clist = [cmap[groups[node]] for node in MST.nodes()]
    
    plt.figure(facecolor = 'w', figsize = (20,20))
    ax = plt.axes()
    
    x_pos = [pos[0] for pos in MST_pos.values()]
    y_pos = [pos[1] for pos in MST_pos.values()]
    
    ax.set_xlim(min(x_pos) * 1.1, max(x_pos) * 1.1)
    ax.set_ylim(min(y_pos) * 1.1, max(y_pos) * 1.1)
    
    nx.draw_networkx_edges(MST, pos = MST_pos, ax = ax, edgelist = diam_edges.edges(), width=width_diam)
    
    nx.draw_networkx(MST, pos = MST_pos, ax = ax, with_labels = False, node_size = node_size, linewidths = linewidths, 
                     width = width, edge_color = edge_color, node_color = clist, vmin = 0, vmax = 1)