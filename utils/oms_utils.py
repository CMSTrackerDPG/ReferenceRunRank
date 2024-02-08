# local modules
import get_oms_data
from get_oms_data import get_oms_api, get_oms_data, get_oms_response_attribute

import pandas as pd
import numpy as np
import json
import time
import refrank_utils as rrr

def subdivide_range(oldest_run, newest_run, step=1000):
    """
    Subdivide a range from x to y into sub-ranges of size 'step'.
    
    Args:
        x (int): Start of the range.
        y (int): End of the range.
        step (int): The size of each sub-range.
        
    Returns:
        list of tuples: A list of tuples, each representing a sub-range.
    """
    ranges = []
    for start in range(oldest_run, newest_run, step):
        end = start + step
        if end > newest_run:
            ranges.append((start, newest_run)) 
        else:
            ranges.append((start, end - 1)) 
    
    return ranges

def get_runs_lss(run_range, omsapi, run_attribs=[], ls_attribs=[], run_filters=[], ls_filters=[]):
    """
    Downloads run and lumisection level metadata for a given range of runs.
    
    Args:
        run_range (tuple): Start and end of range of runs to be loaded
        omsapi: API instance to load data from OMS
        attributes (list): Names of all of the attributes 
    
    Returns:
        Tuple of dataframes: Two pandas dataframes containing the requested run 
        and lumisection level features for the specified range or runs.
    """
    
    run_df = download_oms_data(run_range, omsapi, "runs", run_attribs, extrafilters=run_filters)
    ls_df = download_oms_data(run_range, omsapi, "lumisections", ls_attribs, extrafilters=ls_filters)
    return run_df, ls_df
    
    
    
def download_oms_data(run_range, omsapi, attribs_level, attribs=[], extrafilters=[]):
    """
    Downloads run and lumisection level metadata for a given range of runs.
    
    Args:
        run_range (tuple): Start and end of range of runs to be loaded
        omsapi: API instance to load data from OMS
        attribs_level (string): Describes if the data to be downloaded are run or LS level features.
        attribs (list): Names of all of the attributes 
        subdivs_steps (int): Number of (potential) runs after which the runs will be loaded sequentially instead of all at once.
    
    Returns:
        pd.DataFrame: Dataframe containing all of the requested features for all available runs.
    """
    
    if attribs_level == "runs":
        limit_entries = 5000
        range_limit = limit_entries
    elif attribs_level == "lumisections":
        limit_entries = 100_000
        range_limit = 1000
    else:
        return None
    
    if run_range[1] - run_range[0] > range_limit:
        run_range_list = subdivide_range(run_range[0], run_range[1], range_limit)
        oms_df = pd.DataFrame(columns=attribs)
        for runs in run_range_list:
            oms_json = get_oms_data(
                omsapi, 
                attribs_level, 
                runs, 
                limit_entries=limit_entries,
                attributes=attribs,
                extrafilters=extrafilters
            )

            oms_df = pd.concat([oms_df, rrr.makeDF(oms_json).convert_dtypes()])
            time.sleep(2)
    else: 
        oms_json = get_oms_data(
                omsapi, 
                attribs_level, 
                run_range, 
                limit_entries=limit_entries,
                attributes=attribs,
                extrafilters=extrafilters
            )
        oms_df = rrr.makeDF(oms_json).convert_dtypes()
        
    return oms_df
    