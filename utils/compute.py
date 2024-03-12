"""
This file contains functions that are good for computing extra features not included in the OMS query
"""

def comp_delta_totallumi(run_df):
    """
    Compute the change in total luminosity for each run using run level data.
    
    Args:
        run_df: Dataframe containing each run (target and candidates) with at least its initial and final luminosity.
        
    Returns:
        run_df, but with added delta_totallumi column
    """
    run_df['delta_totallumi'] = run_df['end_lumi'] - run_df['init_lumi']
    
    return run_df

def comp_temp_dist(run_df, target):
    """
    Computer the temporal distance between a given target run and each of the 
    candidate runs and adds these results as a column to the input run dataframe.
    
    Args:
        run_df: Dataframe containing each run (target and candidates) with at least its start time.
        target: Target run which will serve as temporal reference.
        
    Returns:
        run_df, but with added temp_dist column
    """
    
    # Convert data types
    run_df['start_time'] = pd.to_datetime(run_df['start_time'])
    run_df['end_time'] = pd.to_datetime(run_df['end_time'])
    
    try: # Try to get target time
        target_time = run_df[run_df['run_number']==target]['start_time'].item()
    except:
        print("ERROR: Target not found in given dataframe.")
        return run_df
    
    # Getting temporal distance
    run_df['temp_dist'] = (target_time - run_df['start_time']).dt.total_seconds()
    return run_df

def get_pileup(lsdf):
    """
    Gets the pileup from the lumi df and returns a Pandas Series with mean PU per run.
    """
    return lsdf.groupby(['run_number'])['pileup'].mean()

def get_avg_initLumi(lsdf):
    """
    Gets average init luminosity for each run from the LS dataframe
    """
    
    return lsdf.groupby(["run_number"])["init_lumi"].mean()