""" Module contains functions to help generate binary_metric_matrix, diagnose and explore clusters"""

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.metrics import silhouette_samples, silhouette_score
from tqdm.notebook import tqdm

''' Cluster functions'''
name = 'cluster'


def create_binary_metric_matrix(arr, func, dec=2, **args):
    '''
    Generic function to used to generate a similarity/dissimiarity matrix based
    on a binary metric

    Parameters
    ----------

    arr: 2d numpy array
        contains the binary cases to which we want to apply metric `func`
    func: Python function
        binary metric function that accepts a, b, c, d
    dec: integer
        number of decimals for output
    args: dict
        additional arguments we want to pass to `func`

    Return
    ------

    numpy array:
        Matrix that resuls after applying metric `func`

    Notes
    -----
        This function may be used to calculate any binary distance or
        similarity index matrix

    '''

    # determine presence and absence
    present = arr == 1
    absent = arr == 0

    # extend presence and absence for broadcasting
    P = present[:, :, np.newaxis]
    rP = present.T[np.newaxis, :, :]
    A = absent[:, :, np.newaxis]
    rA = absent.T[np.newaxis, :, :]

    # Use broadcasting to calculate P
    P_P = (P & rP)
    P_A = (P & rA)
    A_P = (A & rP)
    A_A = (A & rA)

    A = np.sum(P_P, axis=1)
    B = np.sum(P_A, axis=1)
    C = np.sum(A_P, axis=1)
    D = np.sum(A_A, axis=1)

    return(np.round(func(A, B, C, D, **args), dec))

# CLUSTER PROFILE


def bitprofile(row):
    '''
    Generates a 'bit profile' string representing the profile (presence/
    absence)

    Parameters
    ----------

    row: Pandas series
        Series representing a row in a Pandas dataframe

    Return
    ------

    String
        Cluster profile as a string of 1s and 0s
    '''
    bp = np.where(row.values > 0, 1, 0).astype(str).tolist()

    return ''.join(bp)


def cluster_summaries(data, labels, opt='dec'):
    '''
    Generates summary information for each cluster

    Parameters
    ----------

    data: Pandas dataframe
        Each row represents contains weights/counts
    labels: list
        labels produced by the clustering technique
    opt: string
        if 'dec' (default) returns the profile in a cluster in decimal rather
        than in binary form.

    Return
    ------

    summary_tbl: Pandas dataframe
        contains the following columns:

        - *cluster*: cluster label.
        - *bp*: binary profile.
        - *counts*: count of clusters with this profile.
        - *clst_sig*: cluster signature. Sequence of binary or decimal
                      numbers that make up a cluster signature.


    '''

    # generate summary table
    summary_tbl = (data.assign(bp=data.apply(bitprofile, axis=1))
                   # creates bit profile of each row
                       .assign(cluster=sort_cluster_labels(labels))
                   # creates a column with the cluster labels of each
                   # row (after sorting them)
                       .groupby(['cluster', 'bp']).size())
                   # group by cluster label and bit profile

    # initialize list to hold cluster signatures
    clst_sig = []

    # iterate thru each cluster label
    for c in summary_tbl.index.get_level_values(0):

        # extract all bitprofiles within current cluster
        bps = summary_tbl[c].index.sort_values().tolist()

        # convert to decimal?
        if opt == 'dec':
            bps = [str(int(bp, 2)) for bp in bps]

        # combine bitprofiles to generate cluster signature
        clst_sig += ['-'.join(bps)]

    # flatten table
    summary_tbl = (summary_tbl.reset_index(name='counts')
                              .sort_values(by=['cluster', 'counts'],
                                           ascending=(True, False)))

    # add cluster signature column to summary table
    summary_tbl['clst_sig'] = clst_sig

    return summary_tbl


def sort_cluster_labels(lbls):
    '''
    Sorts cluster labels in descending order

    Parameters
    ----------

    lbls: list
        cluster labels

    Return
    ------

    lbls_sort: list
        Sorted lbls

    Notes
    -----

        Labels are sorted so that clusters with the greatest number of cases
        have smallest numerical labels (descending order)

    '''

    # make a copy of lbls
    lbls_sort = np.copy(lbls)

    # calculate counts in each cluster
    clst_count = np.bincount(lbls)

    # new cluster label
    newc = np.arange(len(clst_count))

    # find cluster labels in descending order
    oldc = np.argsort(clst_count)[::-1]

    for oldc, newc in list(zip(oldc, newc)):
        lbls_sort[lbls == oldc] = newc

    return lbls_sort


def percentiles(series):
    '''
    Returns the percentile rank for each entry

    Parameters
    ----------

    series: Pandas series

    Return
    ------

    Pandas series:
        Each entry in the series represents a percentile

    Notes
    -----

    This function is meant to be apply to each column (axis=1) in a dataframe
    using `apply()`.It calculates the percentile score that corresponds to each
    observation based on the values in the entire column. So a percentile score
    of 23 would mean that the observation is above 23% of the scores in this
    column.
    '''

    from scipy.stats import percentileofscore as pctl
    return series.apply(lambda x: pctl(series, x, 'weak'))


def cluster_profile(data, clst_sel, lbls, map_dict= None, opt='raw'):
    '''
    Displays a cluster profile in various forms

    Parameters
    ---------

    data: Pandas dataframe
        Original data containing only data clustered
    clst_sel: Numpy arrays
        Selection of pandas rows (making up a cluster)
    lbls: list
        contains two strings used as column names
    map_dict: dictionary
        dictionary mapping columns into new categories
    opt: string
        Option used to display profile:

        - 'raw': (default) Displays a histogram of each *non-empty* column
        - 'pctl': Percentile option. Displays a barchart showing distribution
                  by percentile
        - 'map': Mapping option. Maps columns in data into new categories
                 before displaying a histogram of each *non-empty* column

    Return
    ------

    Axis
        Seaborn (matplotlib) axis
    '''

    # internal function
    def _mapping(key, dic):
        ''' Simple function used to map columns into new categories '''
        return dic[key]

    # select productions with at least one entry in cluster
    nonzero = (data.loc[clst_sel, :] != 0.0).any(axis=0)
    # opposite:(data.loc[sel,:] == 0.0).all(axis = 0)

    # filter data
    data = data.loc[clst_sel, nonzero]

    if opt == 'raw':
        # rearrange data
        data = (data.melt()
                    .query(" value > 0")
               )
        data.columns = lbls

        # plot
        fgrid = sns.displot(data, kind='hist', x=lbls[1], col=lbls[0], bins=15,
                            color='red', height=3, aspect=1.5, col_wrap=4,
                            kde=False,
                            facet_kws={'sharex': False, 'sharey': False})
        return fgrid

    elif opt == 'pctl':
        # rearrange data
        data = (data.apply(percentiles)
                    .melt()
                    .query(" value > 0")
                )

        data.columns = lbls
        # plot
        bins =[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        fgrid = sns.displot(data, kind='hist', x=lbls[1], col=lbls[0],
                            bins=bins, color='blue', height=3, aspect=1.5,
                            col_wrap=4, alpha=0.5, shrink=0.7, kde=False,
                            facet_kws={'sharex': False, 'sharey': False})
        return fgrid

    elif opt == 'map':
        data = (data.melt()
                    .query(" value > 0")
               )
        data.columns = lbls
        data = data.query(lbls[1]+ " > 0")
        data[lbls[0]] = data[lbls[0]].apply(_mapping, dic=map_dict)

        # plot
        fgrid = sns.displot(data, kind='hist', x=lbls[1], col=lbls[0],
                            bins=15, color='green', height=3, aspect=1.5,
                            col_wrap=4, kde=False,
                            facet_kws={'sharex': False, 'sharey': False})
        return fgrid


def plot_silhouette(data, range_n_clusters, Clusterer, clst_kw,
                    func_lbl=lambda Clusterer: Clusterer.labels_,
                    matrix_flag=True, size=(7, 26), colrow=(2, 3)):
    '''
    Calculates average silhouette score and plots silhouette plot for a range
    of n clusters

    Parameters
    ----------
    data: numpy array
        original data or dissimilarity/distance matrix
    range_n_clusters: list
        range of number of clusters to consider
    Clusterer: scikit_learn clustering type object
    clst_kw: dictionary
        parameters to be used when executing Clusterer
    func_lbl: Python function
        returns labels after fitting Clusterer to data
    matrix_flag: Boolean
        Use 'True' (default) if *data* is a dissimilarity matrix
    size: tuple
        Size of plot
    colrow: tuple
        Specify number of columns/rows in plot


    Return
    ------
    plot
        Silhouettte plot range_m_clusters

    Notes
    -----
    Based on `Silhouette Plot <https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html>`_
    '''

    # create a plot
    fig, axs = plt.subplots(nrows=colrow[0], ncols=colrow[1], figsize=size,
                            sharex='col', sharey='row', squeeze=True)

    for n_clusters, ax in zip(range_n_clusters, axs.flatten()):
        # set x plot range
        ax.set_xlim([-1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        # update Clusterer parameters
        kw = clst_kw.copy()
        kw['n_clusters'] = n_clusters

        # Initialize the clusterer with n_clusters value and a random generator
        clusterer = Clusterer(**kw)
        clusterer.fit(data)
        cluster_labels = func_lbl(clusterer)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the
        # formed clusters
        if matrix_flag:
            silhouette_avg = silhouette_score(data, cluster_labels, metric='precomputed')
            sample_silhouette_values = silhouette_samples(data, cluster_labels, metric='precomputed')
        else:
            silhouette_avg = silhouette_score(data, cluster_labels)
            sample_silhouette_values = silhouette_samples(data, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax.fill_betweenx(np.arange(y_lower, y_upper),
                             0, ith_cluster_silhouette_values,
                             facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax.set_title("Silhouette plot for "+str(n_clusters)+ ' clusters', fontsize=20)

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        xticks = [-1 ,-0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=14)

    # set xlabel for last axis
    fig.text(0.5, 0.05, 'Silhouette Coefficient', ha='center', fontsize=20)
    fig.text(0.1, 0.5, 'Cluster Labels', va='center', rotation='vertical', fontsize=20)

    plt.show()
