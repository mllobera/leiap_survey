# imports
import pandas as pd
import leiap_survey.temporal as tmp
from tqdm.notebook import tqdm

''' Pottery functions'''
name = 'pottery'


def select_productions(ti_tbl, name='Abbrev', criteria={'ti1_min_overlap': 50},
                       opt=None):
    '''
    selects productions based on criteria

    Parameters
    ----------

    ti_tbl: pandas dataframe
        contains the results of :func:`leiap_survey.temporal.
        generate_temporal_intersection_table()`

    name: string
        name of the column holding ceramic production names
    criteria: dictionary
        contains criteria for ceramic production selection. Critera may be
        specify via the following entries:

        - *ti1_min_overlap*: minimum amount of temporal interval overlap. Ceramic
          productions with this or greater overlap will be selected
        - *ti2__min_overlap* :  minimum amount of ceramic production overlap. Ceramic
          productions with this or greater overlap will be selected
        - *gap*: minimum amount of gap. Ceramic productions with this or
          greater amount of gap will be selected
        - *itype*: a list containing the type of intersections to be selected
          (see `calculate_temporal_intersection()`)

    opt: string
        Use 'table' to generate table output. Leave blank to return only names
        of ceramic productions (default)


    Return
    ------
    tbl: pandas dataframe
        Table with names of diagnostic ceramic productions

'''

    # process criteria text
    if 'ti1_min_overlap' in criteria.keys():
        qury = "ti1_overlap >= @criteria['ti1_min_overlap']"
    else:
        print('\n ERROR: Need ti_overlap!')
        return None

    if 'ti2_min_overlap' in criteria.keys():
        qury += " and ti2_overlap >= @criteria['ti2_min_overlap']"

    if 'gap' in criteria.keys():
        qury += " and gap >= @criteria['gap']"

    if 'itype' in criteria.keys():
        qury += " and itype in @criteria['itype']"

    # select by criteria
    tbl = (ti_tbl.query(qury)
                 .sort_values(['ti1_overlap', 'ti2_overlap', 'gap', 'itype'],
                              ascending=False))

    if opt == 'table':
        return tbl
    else:
        return tbl[name].tolist()


def production_diagnostic_scoring(ti1s_df, ti2, criteria_lst,
                                  ti1s_kw={'start': 'Start',
                                           'end': 'End',
                                           'name': 'Abbrev'}):
    '''
    Scores ceramic productions based on temporal overlap criteria

    Parameters
    ----------

    ti1s_df: pandas dataframe
        contains columns with chronological information (start, end) and name
        (name) of ceramic productions
    ti2: list or numpy array
        time interval for which we want to score entries in *ti1s_df* table
    criteria_lst: list
        each entry is a dictionary containing temporal criteria. Critera may be
        specify via the following entries:

        - *ti1_min_overlap*: minimum amount of temporal overlap with
          entries (=ceramic productions) in ti1s_df so that they are
          selected.
        - *ti2_min_overlap*:  minimum amount of temporal overlap with
          ti2 so that entries (=ceramic productions) in ti1s_df are
          selected.
        - *gap:*: minimum amount of gap. Entries in ti1s_df (= ceramic
          productions) will be selected if their gap with ti2 is equal
          or greater.
        - *itype*: a list containing the type of intersections to be
          selected (see `calculate_temporal_intersection()`)

    ti1s_kw: dictionary
        contains entries (start, end and name) used to identify columns in
        ti1s_df with chronological information (start, end) and production
        names


    Return
    ------

    score_tbl: pandas dataframe
        contains ceramic productions with the count the times that they have
        been selected and a diagnostic score based on frequency

    '''

    # helper function to update scores of ceramic productions
    def _update_score(score, sel):
        for k in sel:
            if k in score.keys():
                score[k] = score[k]+1
            else:
                # initialize when ceramic production is new
                score[k] = 1
        return score

    # iterate over criteria
    score = {}
    for c in tqdm(criteria_lst, desc='scoring...'):

        # generate temporal intersection table
        ti_tbl = tmp.generate_temporal_intersection_table(ti2, ti1s_df, ti1s_kw)

        # select productions that meet criteria
        sel = select_productions(ti_tbl, name=ti1s_kw['name'], criteria=c)

        # update score
        score = _update_score(score, sel)

    # create table with counts and score
    score_tbl = (pd.DataFrame({'production': score.keys(), 'num': score.values()})
                   .assign(score_rel=lambda x: x.num/x.num.max())
                   .assign(score_max=lambda x: x.num/len(criteria_lst))
                   .sort_values(['score_rel', 'score_max'], ascending=False)
                   .reset_index(drop=True))

    return score_tbl
