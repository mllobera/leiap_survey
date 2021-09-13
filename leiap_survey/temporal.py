import pandas as pd
import numpy as np

''' Temporal functions'''
name = 'temporal'


def calculate_temporal_intersection(ti1, ti2):
    '''
    Calculates the intersection between two temporal intervals

    Parameters
    ----------
    ti1: list or numpy array
        temporal interval
    ti2: list or numpy array
        temporal interval/s as a list, list of list, 1D or 2D numpy array

    Return
    ------
    rslt: numpy array
        contains information about the intersection of ti1 with all temporal
        intervals in ti2

    Notes
    -----
        This function provides the following information:
        - intersection1: Temporal intersection described as a percentage
        of delta_ti1
        - intersection2: Temporal intersection described as a percentage
        of delta_ti2
        - gap: temporal gap betweeen ti1 and ti2
        - type: type of intersection according to the following scheme:

        ::

           case 1:
                  ti1:   [       ]
                  ti2:            <--gap-->[     ]

           case 2:
                  ti2:   [       ]
                  ti1:      [ ]

           case 3:
                  ti1:   [       ]
                  ti2:       [     ]

           case 4:
                  ti1:             <--gap-->[     ]
                  ti2:    [       ]

            case 5:
                   ti1:        [ ]
                   ti2:     [       ]

            case 6:
                   ti1:       [     ]
                   ti2:  [       ]

    '''

    if isinstance(ti1, list):
        ti1 = np.array(ti1)
    if isinstance(ti2, list):
        ti2 = np.array(ti2)

    old_ndim = ti2.ndim
    if old_ndim == 1:
        ti2 = np.array([ti2])

    rslt = np.zeros((len(ti2), 4), dtype=np.float32)

    ti1_before_ti2 = ti1[0] <= ti2[:, 0]

    # CASE 1
    ti1_ends_before_ti2 = ti1[1] <= ti2[:, 0]
    sel = ti1_before_ti2 & ti1_ends_before_ti2
    rslt[sel, 0] = 0
    rslt[sel, 1] = 0
    rslt[sel, 2] = ti2[sel, 0]-ti1[1]
    rslt[sel, 3] = 1

    # CASE 2
    ti1_contains_ti2 = ti1[1] >= ti2[:,1]
    sel = ti1_before_ti2 & ti1_contains_ti2
    rslt[sel, 0] = 100 * (ti2[sel, 1]-ti2[sel, 0])/(ti1[1] - ti1[0])
    rslt[sel, 1] = 100
    rslt[sel, 2] = 0
    rslt[sel, 3] = 2

    # CASE 3
    ti1_intersects_ti2 = (ti1[1] >= ti2[:, 0]) & (ti1[1] < ti2[:, 1])
    sel = ti1_before_ti2 & ti1_intersects_ti2
    rslt[sel, 0] = 100 * (ti1[1]-ti2[sel, 0])/(ti1[1] - ti1[0])
    rslt[sel, 1] = 100 * (ti1[1]-ti2[sel, 0])/(ti2[sel, 1] - ti2[sel, 0])
    rslt[sel, 2] = 0
    rslt[sel, 3] = 3

    ti2_before_ti1 = ti2[:, 0] < ti1[0]  # ~ti1_before_ti2

    # CASE 4
    ti2_ends_before_ti1 = ti2[:, 1] <= ti1[0]
    sel = ti2_before_ti1 & ti2_ends_before_ti1
    rslt[sel, 0] = 0
    rslt[sel, 1] = 0
    rslt[sel, 2] = ti1[0] - ti2[sel, 1]
    rslt[sel, 3] = 4

    # CASE 5
    ti2_contains_ti1 = ti2[:, 1] >= ti1[1]
    sel = ti2_before_ti1 & ti2_contains_ti1
    rslt[sel, 0] = 100
    rslt[sel, 1] = 100 * (ti1[1]-ti1[0])/(ti2[sel, 1] - ti2[sel, 0])
    rslt[sel, 2] = 0
    rslt[sel, 3] = 5

    # CASE 6
    ti2_intersects_ti1 = (ti2[:, 1] >= ti1[0]) & (ti2[:, 1] <= ti1[1])
    sel = ti2_before_ti1 & ti2_intersects_ti1
    rslt[sel, 0] = 100 * (ti2[sel, 1]-ti1[0])/(ti1[1] - ti1[0])
    rslt[sel, 1] = 100 * (ti2[sel, 1]-ti1[0])/(ti2[sel, 1] - ti2[sel, 0])
    rslt[sel, 2] = 0
    rslt[sel, 3] = 6

    if old_ndim == 1:
        return rslt[0]
    else:
        return rslt


def calculate_temporal_intersection_ratio(ti1, ti2):
    '''
    Calculates the intersection between two temporal intervals as a
    ratio

    Parameters
    ----------
    ti1: list or numpy array
        temporal interval
    ti2: list or numpy array
        temporal interval/s as a list, list of list, 1D or 2D numpy array

    Return
    ------
    rslt: numpy array
         contains information about the intersection of ti1 with all temporal
         intervals in ti2

    Notes
    -----
        This function provides the following information:

        - intersection: Temporal intersection described as the
          ratio intersection/delta_ti1
        - gap: temporal gap betweeen ti1 and ti2
        - type: type of intersection according to the following scheme:

        ::

           case 1:
                  ti1:   [       ]
                  ti2:            <--gap-->[     ]

           case 2:
                  ti2:   [       ]
                  ti1:      [ ]

           case 3:
                  ti1:   [       ]
                  ti2:       [     ]

           case 4:
                  ti1:             <--gap-->[     ]
                  ti2:    [       ]

            case 5:
                   ti1:        [ ]
                   ti2:     [       ]

            case 6:
                   ti1:       [     ]
                   ti2:  [       ]

    '''


    old_ndim = ti2.ndim
    if old_ndim == 1:
        ti2 = np.array([ti2])

    rslt = np.zeros((len(ti2), 3), dtype=np.float32)

    ti1_before_ti2 = ti1[0] <= ti2[:, 0]

    # CASE 1
    ti1_ends_before_ti2 = ti1[1] <= ti2[:, 0]
    sel = ti1_before_ti2 & ti1_ends_before_ti2
    rslt[sel, 0] = 0
    rslt[sel, 1] = ti2[sel, 0]-ti1[1]
    rslt[sel, 2] = 1

    # CASE 2
    ti1_contains_ti2 = ti1[1] >= ti2[:, 1]
    sel = ti1_before_ti2 & ti1_contains_ti2
    rslt[sel, 0] = (ti2[sel, 1]-ti2[sel, 0])/(ti1[1] - ti1[0])
    rslt[sel, 1] = 0
    rslt[sel, 2] = 2

    # CASE 3
    ti1_intersects_ti2 = (ti1[1] >= ti2[:, 0]) & (ti1[1] < ti2[:, 1])
    sel = ti1_before_ti2 & ti1_intersects_ti2
    rslt[sel, 0] = (ti1[1]-ti2[sel, 0])/(ti1[1] - ti1[0])
    rslt[sel, 1] = 0
    rslt[sel, 2] = 3

    ti2_before_ti1 = ti1_before_ti2  # ti2[:,0] < ti1[0]

    # CASE 4
    ti2_ends_before_ti1 = ti2[:, 1] <= ti1[0]
    sel = ti2_before_ti1 & ti2_ends_before_ti1
    rslt[sel, 0] = 0
    rslt[sel, 1] = ti1[0] - ti2[sel, 1]
    rslt[sel, 2] = 4

    # CASE 5
    ti2_contains_ti1 = ti2[:, 1] >= ti1[1]
    sel = ti2_before_ti1 & ti2_contains_ti1
    rslt[sel, 0] = (ti2[sel, 1]-ti2[sel, 0])/(ti1[1] - ti1[0])
    rslt[sel, 1] = 0
    rslt[sel, 2] = 5

    # CASE 6
    ti2_intersects_ti1 = (ti2[:, 1] >= ti1[0]) & (ti2[:, 1] <= ti1[1])
    sel = ti2_before_ti1 & ti2_intersects_ti1
    rslt[sel, 0] = (ti2[sel, 1]-ti1[0])/(ti1[1] - ti1[0])
    rslt[sel, 1] = 0
    rslt[sel, 2] = 6

    if old_ndim == 1:
        return rslt[0]
    else: 
        return rslt


def generate_temporal_intersection_table(ti1, ti2_df,
                                         ti2_kw={'start': 'Start',
                                                 'end': 'End',
                                                 'name': 'Abbrev'}):
    '''
    Creates table of temporal intersections

    Parameters
    ----------
    ti1: list or numpy array
        temporal interval
    ti2_df: Pandas dataframe
        temporal interval/s as dataframe
    ti2_kw: dictionary
        contains information on what are the columns with start, end, and name

    Returns
    -------
    tbl: Pandas dataframe
        contains information about the intersection of ti1 with all temporal
        intervals in ti2_df
    '''


    if isinstance(ti1, list):
        ti1 = np.array(ti1)

    start = ti2_kw['start']
    end = ti2_kw['end']
    name = ti2_kw['name']

    # return numpy array
    ti2 = ti2_df[[start, end]].to_numpy()

    data = calculate_temporal_intersection(ti1,ti2)
    tbl = pd.DataFrame(data, columns=['ti1_overlap', 'ti2_overlap',
                       'gap', 'itype'])
    tbl[name] = ti2_df[name].to_list()

    return tbl


def create_time_intervals(start, end, interval):
    '''
    Generates time intervals

    Parameters
    ----------
    start: integer
        Use negative values for BCE dates and positive for CE
    end: integer
        Use negative values for BCE dates and positive for CE
    interval: integer
        Temporal interva

    Return
    ------
        dataframe where each row is a time interval.

    Notes
    -----
        This function will adjust one or both endpoints in order to generate a ¡
        even distribution based on the interval. If time periods span from BCE
        to CE it will adjust endpoints in order to have time periods that start
        and end with change in era.
    '''


    # check right entries
    if start > end-interval:
        raise Exception(f'Start must be larger than End by {interval} ')

    # generate time intervals (includes both ends)
#    time_intervals = np.arange(start, end+interval, interval)
    if start < 0:
        if end <= 0:
            # adjust start
            adj_start = (start // interval) * interval
            time_intervals = np.arange(end, adj_start, -interval)[::-1]
        else:  # end > 0
            # adjust both
            adj_start = (start // interval) * interval
            adj_end = (end // interval) * interval
            time_intervals = np.arange(adj_start, adj_end+interval, interval)
    else:  # start >= 0:
        if end > 0:
            # adjust end
            adj_start = start
            adj_end = (end // interval) * interval
            time_intervals = np.arange(adj_start, adj_end+interval, interval)

    end_interval = time_intervals[1:]
    start_interval = time_intervals[:-1]

    # generate labels
    ti_labels = ['ti'+str(n) for n in range(len(start_interval))]

    # generate df via dictionary
    tis = {t: {'start': s, 'end': e}
           for t, s, e in zip(ti_labels, start_interval, end_interval)}

    return pd.DataFrame(tis).T.loc[:, ['start', 'end']]


def ao_find_temporal_intersection(ti1, ti2):
    '''
    Finds temporal intersection between time intervals for aoristic analysis

    Parameters
    ----------
    ti1: list or numpy array
        temporal interval
    ti2: list or numpy array
        temporal interval/s as a list, list of list, 1D or 2D numpy array

    Return
    ------
    rslt: numpy array
        contains intersection of ti1 with ti2

    Notes
    -----
        Intersection is described as a ratio of delta_ti2 or temporal
        intersection with over delta_ti1
    '''


    if isinstance(ti1, list):
        ti1 = np.array(ti1)

    if isinstance(ti2, list):
        ti2 = np.array(ti2)

    old_ndim = ti2.ndim

    if old_ndim == 1:
        ti2 = np.array([ti2])

    rslt = np.zeros(len(ti2), dtype=np.float32)

    # calculate delta_time (time length)
    delta_ti1 = ti1[1]-ti1[0]
    delta_ti2 = ti2[0, 1]-ti2[0, 0]

    # CASE 1
    ti1_starts_before_ti2_starts = ti1[0] <= ti2[:, 0]

    # CASE 1a
    ti1_ends_before_ti2_starts = ti1[1] <= ti2[:, 0]
    sel = ti1_starts_before_ti2_starts & ti1_ends_before_ti2_starts
    rslt[sel] = 0.0

    # CASE 1b
    ti1_ends_after_ti2_ends = ti1[1] >= ti2[:, 1]
    sel = ti1_starts_before_ti2_starts & ti1_ends_after_ti2_ends
    rslt[sel] = delta_ti2 / delta_ti1

    # CASE 1c
    ti1_ends_within_ti2 = (ti1[1] >= ti2[:, 0]) & (ti1[1] < ti2[:, 1])
    sel = ti1_starts_before_ti2_starts & ti1_ends_within_ti2
    rslt[sel] = (ti1[1]-ti2[sel, 0])/delta_ti1  # (ti2[sel,1] - ti2[sel,0])

    # CASE 2
    ti2_starts_before_ti1_starts = ti2[:, 0] < ti1[0]  # ~ti1_before_ti2

    # CASE 2a
    ti2_ends_before_ti1_starts = ti2[:,1] <= ti1[0]
    sel = ti2_starts_before_ti1_starts & ti2_ends_before_ti1_starts
    rslt[sel] = 0.0

    # CASE 2b
    ti2_ends_after_ti1_ends = ti2[:, 1] >= ti1[1]
    sel = ti2_starts_before_ti1_starts & ti2_ends_after_ti1_ends
    rslt[sel] = delta_ti2 / delta_ti1

    # CASE 2c
    ti2_ends_within_ti1 = (ti2[:, 1] >= ti1[0]) & (ti2[:, 1] <= ti1[1])
    sel = ti2_starts_before_ti1_starts & ti2_ends_within_ti1
    rslt[sel] = (ti2[sel, 1] - ti1[0]) / delta_ti1

    if old_ndim == 1:
        return rslt[0]
    else:
        return rslt


def ao_calculate_weights(df, df_ti, chrono=['Start', 'End']):
    '''
    Calculates aoristic weights

    Parameters
    ----------
    df: Pandas dataframe
        Dataframe where each row (usually a feature) is associated with a
        temporal interval
    df_ti: Pandas dataframe
        Each row represents a temporal interval (usually the output from
        ```create_temporal_intersections()```)
    chrono: list
        contains the name of the columns in df_ti that define the temporal
        interval

    Return
    ------
    Pandas dataframe
        Dataframe with aoristic weights (intersections) for each row and time interval
        combination
    '''
    rslt = []
    for ti in df[chrono].values.tolist():
        rslt += [ao_find_temporal_intersection(ti, df_ti.values.tolist())]
    return pd.DataFrame(data=rslt, columns=df_ti.index)


def ao_calculate_probability_from_weights(feature, aow):
    '''
    Calculates probabilities based on aoristic weights

    Parameters
    ----------
    feature: Pandas dataframe
        Contains one or several rows with the same feature identifier
    aow: Pandas dataframe
        Contains aoristic weights for each row in feature

    Return
    ------
    Pandas dataframe
        Dataframe with the probability for distinct feature and temporal interval
        combination

    Notes
    -----
    Aoristic weights represent the probability that a feature is present at a
    time interval. A feature (e.g. site, field, survey point) will typically
    be associated with one or more temporal entries in the aoristic weight
    dataframe. The probabiliy that a feature occured at any time interval is
    the probability that at least one of these temporal entries (row) occurred.

    Example
    -------
    +--------------+-----+-----+-----+---+-----+
    |              | ti1 | ti2 | ti3 |...| tn  |
    +==============+=====+=====+=====+===+=====+
    | Feat1 Entry1 | 0.2 |     |     |   |     |
    +--------------+-----+-----+-----+---+-----+
    | Feat1 Entry2 | 0.1 |     |     |   |     |
    +--------------+-----+-----+-----+---+-----+

    .. math::

       \mathsf {P(Feat1 / ti1) = P(Entry1 \cap Entry2) \cup P(\lnot Entry1 \cap Entry2) \cup P(Entry1 \cap \lnot Entry2)=

       = P(Entry1) \cdot P(Entry2) + P(\lnot Entry1) \cdot P(Entry2) + P(Entry1) \cdot P( \lnot Entry2)=

       = 0.2 \cdot 0.1 + 0.8 \cdot 0.1 + 0.2 \cdot 0.9 = 0.28

       \\therefore P(Feat1 / ti1) = 1 - P(\lnot Entry1 \cap \lnot Entry2) = 1 - (0.8 \cdot 0.9) = 0.28 }

    .. note::
       NB1. Feature identifies should be indexes of feature dataframe

    .. note::
       NB2. A factor of 1e6 is used to make sure that probability across each
       feature adds up to less than 1.0 (for the sake of simulation)


    '''


    # extract temporal interval columns labels
    ti_cols = aow.columns.values.tolist()

    # extract feature identifiers
    feat_cols = feature.index.names

    # concatenate features with their aoristic weights
    tmp = pd.concat([feature.reset_index(feat_cols),aow], axis='columns')

    # function used to calculate probability for each feature from the
    # probabilites of all rows associated with that feature
    def not_p(series):
        p_not = 1.0 - series
        return 1 - np.prod(p_not)

    agg_dict = dict.fromkeys(ti_cols, not_p)

    # group by unique feature
    grp_tmp = tmp.groupby(feat_cols, as_index=feature.index.names).agg(agg_dict)

    return grp_tmp.divide(grp_tmp.sum(axis='columns') + 0.000001, axis='rows')


def ao_simulation(prob_feat, N=1000):
    '''
    Simulates presence of a feature at a time period

    Parameters
    ----------
    prob_feat: Pandas dataframe
        A dataframe where each row represents a different feature and each
        column represents the probability of the presence of the feature in
        each time period (=column)
    N: Integer
        Number of simulations

    Return
    ------
        dataframe with N rows where each row has the number of features present at each time period (=column)
    '''
    from scipy.stats import multinomial
    from tqdm.notebook import tqdm

    sim = []
    for n in tqdm(range(N), desc='Simulation...'):
        all_runs = []
        for p in prob_feat.values:
            run = multinomial.rvs(n=1, p=p, size=1)
            all_runs += run.tolist()
        # add each time interval
        sim += [np.sum(all_runs, axis=0).tolist()]
    # create dataframe with results
    return pd.DataFrame(data=sim, columns=prob_feat.columns)
