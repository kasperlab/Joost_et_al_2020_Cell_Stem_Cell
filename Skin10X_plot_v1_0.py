################################################################################
############################## 10X -- PLOT #####################################
################################################################################

"""
Scripts for drawing.
Version: Python 3
"""

################################################################################
################################ DEPENDENCIES ##################################
################################################################################

import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

################################################################################
############################### MISC FUNCTIONS #################################
################################################################################

def remove_ticks(axes, linewidth = 0.5):
    """
    Removes ticks from matplotlib Axes instance
    """
    axes.set_xticklabels([]), axes.set_yticklabels([])
    axes.set_xticks([]), axes.set_yticks([])
    for axis in ['top','bottom','left','right']:
        axes.spines[axis].set_linewidth(linewidth)
        
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

def clean_axis(ax):
    """Remove ticks, tick labels, and frame from axis"""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


################################################################################
################################## PLOTTING ####################################
################################################################################


def draw_log2_var_polyfit(dataset, log2_var_diff, z, selected):
    
    """
    Draws feature selection based on 2nd polynomal fit of log2 means x log2 variance.
    Called in select_features_log2_var_polyfit_v2.
    """
    
    data_mean = dataset.mean(axis = 1)
    data_var = dataset.var(axis = 1) 
    
    log2_mean = np.log2(data_mean + 1)
    log2_var = np.log2(data_var + 1)
    
    line_x = np.arange(log2_mean.min(), log2_mean.max(), 0.01)
    line_y = [z[0] * (x**2) + z[1] * x + z[2] for x in line_x]
    
    clist = pd.Series('blue', index = log2_mean.index)
    clist[log2_var_diff[log2_var_diff > 0].index] = 'red'
    
    if np.all(selected != None):
        clist[selected] = 'green'
        
    plt.figure(figsize = [10,10], facecolor = 'w')
    ax0 = plt.axes()
    
    ax0.set_xlabel('Mean number of transcripts [log2]')
    ax0.set_ylabel('Variance [log2]')
    
    ax0.set_xlim(log2_mean.min() - 0.5, log2_mean.max() + 0.5)
    ax0.set_ylim(log2_var.min() - 0.5, log2_var.max() + 0.5)
    
    ax0.scatter(log2_mean, log2_var, c = clist, linewidth = 0,)
    ax0.plot(line_x, line_y, c = 'r', linewidth = 3)
    
################################################################################

def draw_heatmap(dataset, cell_groups, gene_groups, cmap = plt.cm.jet):
    
    """
    Draw heatmap showing gene expression ordered according to cell_groups and
    gene_groups Series (e.g. AP clustering). Cell and gene groups membership is 
    visualized in two additional panels:
    ----------
    dataset: pd.DataFrame of m cells * n genes.
    cell_groups: pd.Series with ordered cluster identity of m cells.
    gene_groups: pd.Series with ordered cluster identity of n genes.
    cmap: matplotlib color map. Default: plt.cm.jet.
    """
    
    dataset = dataset.ix[gene_groups.index, cell_groups.index]
    dataset = dataset.apply(lambda x: x / max(x), axis = 1)

    plt.figure(figsize=(20,20), facecolor = 'w')
    
    #draw heatmap

    axSPIN1 = plt.axes()
    axSPIN1.set_position([0.05, 0.05, 0.9, 0.9])
    
    axSPIN1.imshow(dataset, aspect = 'auto', interpolation = 'nearest', cmap = cmap)
    
    remove_ticks(axSPIN1)
    
    #draw genes bar

    divider = make_axes_locatable(axSPIN1)

    axGene_gr = divider.append_axes("right", size= 0.5, pad=0.05)

    axGene_gr.set_xlim(-0.5,0.5)
    axGene_gr.imshow(np.matrix(gene_groups).T, aspect = 'auto')
    
    remove_ticks(axGene_gr)
    
    #draw genes bar ticks
    
    gene_groups_ticks = pd.Series(index = set(gene_groups))
    
    for gr in gene_groups_ticks.index:
                
        first_ix = list(gene_groups.values).index(gr)
        last_ix = len(gene_groups) - list(gene_groups.values)[::-1].index(gr)
        gene_groups_ticks[gr] = first_ix + ((last_ix - first_ix) / 2.0)
        
    axGene_gr.set_yticks(gene_groups_ticks.values)
    axGene_gr.set_yticklabels(gene_groups_ticks.index)
    axGene_gr.yaxis.set_ticks_position('right')
    
    #draw cells bar
    
    axCell_gr = divider.append_axes("bottom", size= 0.5, pad=0.05)

    axCell_gr.set_ylim(-0.5, 0.5)
    axCell_gr.imshow(np.matrix(cell_groups), aspect = 'auto')
    
    remove_ticks(axCell_gr)
    
    #draw cells bar ticks
    
    cell_groups_ticks = pd.Series(index = set(cell_groups))
        
    for gr in cell_groups_ticks.index:
                
        first_ix = list(cell_groups.values).index(gr)
        last_ix = len(cell_groups) - list(cell_groups.values)[::-1].index(gr)
        cell_groups_ticks[gr] = first_ix + ((last_ix - first_ix) / 2.0)
        
    axCell_gr.set_xticks(cell_groups_ticks.values)
    axCell_gr.set_xticklabels(cell_groups_ticks.index)
    axCell_gr.xaxis.set_ticks_position('bottom')

################################################################################

def draw_AP_dist_mat(dist_mat, groups, **kwargs):
    
    """
    Draws distance matrices of either m cells or n genes randomly shuffled and
    ordered according to group Series (e.g. AP clustering).
    ----------
    dist_mat: pd.DataFrame with distances of either m x m cells or n x n genes.
    groups: pd.Series with ordered cluster identity of m cells or n genes.
    """
    
    plt.figure(figsize = [20,10], facecolor = 'w')

    ax0 = plt.subplot(121)

    tmp_ix = list(dist_mat.index)
    random.shuffle(tmp_ix)

    ax0.matshow(dist_mat.ix[tmp_ix, tmp_ix], **kwargs)

    ax1 = plt.subplot(122)

    ax1.matshow(dist_mat.ix[groups.index, groups.index], **kwargs)
    
################################################################################
    
def draw_tSNE(tsne_coords, cell_groups, number = None, cmap = plt.cm.jet, pad = 0):
    
    """
    Function to draw tSNE plots.
    ----------
    tsne_coords: DataFrame of tSNE coordinates in two dimensions.
    cell_groups: Series of cell group identity.
    number: int for indentification of plot in tSNE screen.
    """
    
    #initialize figure

    height = 14
    width = 14

    plt.figure(facecolor = 'w', figsize = (width, height))

    #define x- and y-limits

    x_min, x_max = np.min(tsne_coords['x']), np.max(tsne_coords['x'])
    y_min, y_max = np.min(tsne_coords['y']), np.max(tsne_coords['y'])
    x_diff, y_diff = x_max - x_min, y_max - y_min

    if x_diff > y_diff:
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min * (x_diff / y_diff) - pad, y_max * (x_diff / y_diff) + pad)

    if x_diff < y_diff:
        xlim = (x_min * (y_diff/x_diff) - pad, x_max * (y_diff/x_diff) + pad)
        ylim = (y_min - pad, y_max + pad)

    
    #define x- and y-axes

    ax1 = plt.subplot()

    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])

    remove_ticks(ax1)
    
    #define colormap
    
    if type(cmap) != dict:
        cm = cmap
        cmap = {}
        for ix, gr in enumerate(return_unique(cell_groups)):
            cmap[gr] = cm(float(ix) / len(set(cell_groups)))
            
    clist_tsne = [cmap[cell_groups[c]] for c in cell_groups.index]

    ax1.scatter(tsne_coords.ix[cell_groups.index, 'x'],
                tsne_coords.ix[cell_groups.index, 'y'], 
                s = 100,
                linewidth = 0.0,
                c = clist_tsne)
    
    #draw number
    
    ax1.text(xlim[1] * 0.9, ylim[1] * 0.9, number, family = 'Arial', fontsize = 25)
    
    
################################################################################
    
def draw_barplots_v2(dataset, cell_groups, genes, cmap = plt.cm.jet):
    
    """
    draws expression of selected genes in order barplot with juxtaposed group identity
    -----------
    dataset: pd.DataFrame of n samples over m genes
    sample_group_labels: ordered (!) pd.Series showing sample specific group indentity 
    list_of_genes: list of selected genes
    color: matplotlib cmap
    """
    
    # set figure framework
    
    plt.figure(facecolor = 'w', figsize = (21, len(genes) * 3 + 1))
        
    gs0 = plt.GridSpec(1,1, left = 0.05, right = 0.95, top = 1 - 0.05 / len(genes),
                       bottom = 1 - 0.15 / len(genes), hspace = 0.0, wspace = 0.0)
    
    gs1 = plt.GridSpec(len(genes), 1, hspace = 0.05, wspace = 0.0, left = 0.05, right = 0.95, 
                       top = 1 - 0.2 / len(genes) , bottom = 0.05)
    
    #define dataset
    
    dataset = dataset.ix[genes, cell_groups.index]
    
    #define max group ID for color definition
    
    val_max = float(len(return_unique(cell_groups)))
    
    #draw genes
    
    for ix, g in enumerate(genes):
    
        ax = plt.subplot(gs1[ix])
        ax.set_xlim(left = 0, right = (len(dataset.columns)))
                     
        if g != genes[-1]:
            ax.xaxis.set_ticklabels([])
        
        elif g == genes[-1]:
            for tick in ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(15) 
                
        ax.set_ylabel(g, fontsize = 25)
        ax.yaxis.labelpad = 10
        
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
            tick.label.set_label(10)
            
        ax.bar(np.arange(0, len(dataset.columns),1), 
               dataset.ix[g],
               width=1,
               color=[cmap(return_unique(cell_groups).index(val)/val_max) for val in cell_groups.values],
               linewidth=0)
    
    #create groups bar
    
    ax_bar = plt.subplot(gs0[0])
    
    ax_bar.set_xlim(left = 0, right = (len(dataset.columns)))
    ax_bar.set_ylim(bottom = 0, top = 1)
    
    for ix, val in enumerate(cell_groups.values):
        
        ax_bar.axvspan(ix,
                       ix+1,
                       ymin=0,
                       ymax=1, 
                       color = cmap(return_unique(cell_groups).index(val) / val_max))
        
    remove_ticks(ax_bar)
    
################################################################################
    
def draw_barplots_QC(data,param,plate,cmap,**kwargs):

    """
    Draws barplots showing number of reads or unique genes per cell.
    ----------
    data: pd.DataFrame of m cells x n genes.
    param: whether to plot sum of [reads] or sum of unique [genes].
    plate: pd.Series containing plate ID for each cell or other metadata parameter.
    cmap: colormap [dict] for plate.
    """
    
    #compile data based on param
    
    if param == 'reads': data = data.sum(axis=0)
    elif param == 'genes': data = (data>0).sum(axis=0)
        
    #plot data
    
    #initialize figure

    height = 10
    width = 20
    plt.figure(facecolor = 'w', figsize = (width, height))
    
    ax = plt.subplot(111)
    ax.set_xlim(0, len(data.index))
    if param == 'reads': ax.set_ylabel('Number of reads per cell')
    elif param == 'genes': ax.set_ylabel('Number of genes per cell')
        
    #define patient colorscheme
    
    clist = [cmap[plate[ix]] for ix in data.index]
    
    #plot bars
    
    ax.bar(range(len(data.index)),
           data, color = clist,
           width = 1.0,
           **kwargs)
    
    #plot median and mean
    
    ax.axhline(np.median(data), color = 'blue')
    ax.axhline(np.mean(data), color = 'red')

################################################################################

def draw_scatter_groups(coords, groups, cmap = plt.cm.tab20, pad = 2, s = 50):

    """
    Draws scatterplot colored accoring to discrete group indentity.
    ----------
    coords: np.array of 2-dimensional coordinates in m cells.
    groups: discrete group identities for m cells.
    cmap: cmap [dict] linked to groups.
    """
    
    #initialize figure

    height = 10
    width = 13

    plt.figure(facecolor = 'w', figsize = (width, height))
    gs = plt.GridSpec(1,2, wspace=0.025, width_ratios=[10,3])

    #define x- and y-limits

    x_min, x_max = np.min(coords[:,0]), np.max(coords[:,0])
    y_min, y_max = np.min(coords[:,1]), np.max(coords[:,1])
    x_diff, y_diff = x_max - x_min, y_max - y_min
    x_cent, y_cent = x_min + 0.5 * x_diff, y_min + 0.5 * y_diff,

    pad = pad

    if x_diff > y_diff:
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_cent - 0.5 * x_diff - pad, y_cent + 0.5 * x_diff + pad,)

    if x_diff < y_diff:
        xlim = (x_cent - 0.5 * y_diff - pad, x_cent + 0.5 * y_diff + pad,)
        ylim = (y_min - pad, y_max + pad)

    #define x- and y-axes

    ax = plt.subplot(gs[0])

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    #define colormap
    
    if type(cmap) != dict:
        cm = cmap
        cmap = {}
        for ix, gr in enumerate(return_unique(groups)):
            cmap[gr] = cm(float(ix) / 20)
            
    clist = [cmap[groups[c]] for c in groups.index]
    
    #plot

    ax.scatter(coords[:,0],
               coords[:,1], 
               s = s,
               linewidth = 0.0,
               c = clist)
    
    #plot legend
    
    grs = list(set(groups))
    grs.sort()
    
    ax = plt.subplot(gs[1])
    ax.set_xlim(0,1)
    if len(grs) <= 10: ax.set_ylim(10.5, -0.5)
    else: ax.set_ylim(len(grs) + 0.5, -0.5)
        
    for p, i in enumerate(grs):
        ax.scatter(0.15, p, color = cmap[i], s = 200)
        ax.text(0.3, p, i, fontsize = 25, family = 'Arial', va = 'center')
        
    clean_axis(ax)

################################################################################

def draw_scatter_expr(coords, expr, vmin, vmax,text = None,  cmap = plt.cm.viridis, pad = 2, s = 50):

    """
    Draws scatterplot colored according to continuous variable (e.g. gene expression). 
    ----------
    coords: np.array of 2-dimensional coordinates in m cells.
    express: array or list of continous values for m cells.
    vmin: minimum value
    vmax: maximum value
    """
    
    if vmax < 1: 
        vmax = 1
    
    #initialize figure
    
    height = 10
    width = 10.5
    plt.figure(facecolor = 'w', figsize = (width, height))
    gs = plt.GridSpec(1,2, wspace=0.025, width_ratios=[10,.5])

    #define x- and y-limits

    x_min, x_max = np.min(coords[:,0]), np.max(coords[:,0])
    y_min, y_max = np.min(coords[:,1]), np.max(coords[:,1])
    x_diff, y_diff = x_max - x_min, y_max - y_min
    x_cent, y_cent = x_min + 0.5 * x_diff, y_min + 0.5 * y_diff,

    pad = pad

    if x_diff > y_diff:
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_cent - 0.5 * x_diff - pad, y_cent + 0.5 * x_diff + pad,)

    if x_diff < y_diff:
        xlim = (x_cent - 0.5 * y_diff - pad, x_cent + 0.5 * y_diff + pad,)
        ylim = (y_min - pad, y_max + pad)

    #define x- and y-axes

    ax = plt.subplot(gs[0])

    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    
    #define colormap
            
    clist = [cmap((e-vmin)/(vmax-vmin)) for e in expr]
        
    #plot

    ax.scatter(coords[:,0],
               coords[:,1], 
               s = s,
               linewidth = 0.0,
               c = clist)
    
    #text
    
    if text:
        ax.text(xlim[0] + (xlim[1]-xlim[0]) / 2,
                ylim[1] * 1.05,
                text,
                fontsize = 30, va = 'center', ha = 'center')
    
    #plot colorbar
    
    ax = plt.subplot(gs[1])
    
    ax.set_xlim(0,1)
    ax.set_xticks([])
    
    ax.set_ylim(vmin, vmax)
    ax.yaxis.set_ticks_position('right')
    
    for i in np.linspace(vmin, vmax, 100):
        ax.axhspan(i, i + (vmax-vmin) / 100, color = cmap((i-vmin)/(vmax-vmin)))

################################################################################

def draw_scatter_pseudotime(tsne_coords, pseudotime, cmap = plt.cm.viridis, pad = 0):

    """
    Draws scatterplot colored according to cpseudotime. 
    ----------
    coords: pd.DataFrame of ['x'] and ['y'] coordinates in m cells.
    express: array or list of pseudotime values for m cells.
    """
    
    #initialize figure

    height = 14
    width = 14

    plt.figure(facecolor = 'w', figsize = (width, height))

    #define x- and y-limits

    x_min, x_max = np.min(tsne_coords['x']), np.max(tsne_coords['x'])
    y_min, y_max = np.min(tsne_coords['y']), np.max(tsne_coords['y'])
    x_diff, y_diff = x_max - x_min, y_max - y_min

    if x_diff > y_diff:
        xlim = (x_min - pad, x_max + pad)
        ylim = (y_min * (x_diff / y_diff) - pad, y_max * (x_diff / y_diff) + pad)

    if x_diff < y_diff:
        xlim = (x_min * (y_diff/x_diff) - pad, x_max * (y_diff/x_diff) + pad)
        ylim = (y_min - pad, y_max + pad)

    
    #define x- and y-axes

    ax1 = plt.subplot()

    ax1.set_xlim(xlim[0], xlim[1])
    ax1.set_ylim(ylim[0], ylim[1])

    remove_ticks(ax1)
    
    cmin = pseudotime.min()
    cmax = pseudotime.max()
    
    for i in tsne_coords.index:
        
        if i in pseudotime.index:
            ax1.scatter(tsne_coords.ix[i, 'x'],
                        tsne_coords.ix[i, 'y'], 
                        s = 100,
                        linewidth = 0.0,
                        c = cmap((pseudotime[i] - cmin)/(cmax - cmin)))
            
        else:
            ax1.scatter(tsne_coords.ix[i, 'x'],
                        tsne_coords.ix[i, 'y'], 
                        s = 100,
                        linewidth = 0.0,
                        c = 'silver')

################################################################################

def velocyto_plot_pca_2d(vlm, comps = [0,1]):
    
    #generate df
    
    df = pd.DataFrame(index = vlm.ca['CellID'], columns = ['x','y'])
    
    df['x'], df['y'] = vlm.pcs[:,comps[0]], vlm.pcs[:,comps[1]]
    
    #initialize figure

    height = 5
    width = 5

    plt.figure(facecolor = 'w', figsize = (width, height))
    
    ax = plt.subplot(111)
    
    #plot
    
    ax.scatter(df['x'], df['y'], 
               s = 50, 
               c = [vlm.cluster_colors_dict[c] for c in vlm.cluster_labels])