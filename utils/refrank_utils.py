import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def get_weights(df, standardize=True, plot=False):
    """
    Returns the weights for each of the features present in the data contained in a pandas dataframe as determined by the coefficients of the first principal component.
    
    Args:
        df: Dataframe containing each run (target and candidates) with its features
        standardize: Whether or not the features will be standardized.
        plot: Whether or not to make a plot of all the features. Helps with understanding correlation between them.
        
    Returns:
        Returns a numpy array of the weights of the features, in the same order as they appear in the columns of the dataframe.
    """
    if standardize:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    if plot:
        sns.pairplot(df)
        plt.show()
    pca = PCA(n_components=1)
    pca.fit(df)
    pc1_loadings = pca.components_[0]
    pc1_df = pd.DataFrame(pc1_loadings, index=df.columns, columns=['PC1 Loadings'])
    pc1_df['weights'] = pc1_df['PC1 Loadings'] ** 2 # Already normalized
    
    return pc1_df['weights'].to_numpy()


def refrank_pca(features, target, standardize=True, n_components=None):#, keep_original=False):
    """
    Takes in a dataframe of features and ranks each point in feature space by their closseness to the target point. Distance measured in the n-th dimensional PCA sub-space using Eucledian distance. Note that only runs before the target run are considered for ranking.
    
    Args:
        features: Dataframe containing runs along with the features to be considered during ranking. Must include an "run" index column which is used to distinguish each datapoint.
        target: The run that we want to compare other runs to to find potentially good reference runs
        n_components: Number of dimensions of the PCA sub-space (i.e. number of PCA axis used)
        keep_original: Whether or not to keep the original (un-standardized) features as part of the output dataframe. For performance evaluation.
        
    Returns:
        Input dataframe, ordered by rank, with an added column(s) showing the PC component(s).
        
    """
    
    # Filtering out runs that happened before the target run
    features = features.loc[:target]
    
    if len(features) == 1:
        print("ERROR: Not enough runs to perform ranking")
        return None
    
#     if keep_original:
#         features_original = features.copy(deep=True)
        
    run_nums = features.reset_index()['run']
    
    if standardize:
        scaler = StandardScaler()
        features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
        
    # If n_components not, specified, set it to max.
    if n_components is None:
        n_components = len(features.columns)
        
    # Finding PCs
    pca = PCA(n_components=n_components)
    pca.fit(features)
    
    # Constructing dataframe with standardized features
    features_PC = pd.DataFrame(pca.transform(features), columns=['PC'+str(k+1) for k in range(len(pca.components_))])
    features_PC = pd.concat([run_nums, features_PC], axis=1).set_index('run')
    
    # Computing Eucledian distance using coordinates in PCA sub-space and sorting by distance to target run
    dist = np.sqrt(((features_PC - features_PC.loc[target])**2).sum(axis=1))
    features_PC = pd.concat([pd.DataFrame(dist), features_PC], axis=1)
    features_PC.rename(columns = {0:'dist'}, inplace=True)
    features_PC = features_PC.sort_values(by='dist', ascending=True).reset_index()
    
#     if keep_original: 
#         features_PC = pd.merge(features_PC, features_original, left_index=True, right_index=True, how='left')
    
    return features_PC

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

def comp_duration(run_df):
    """
    Compute the duration of each run in seconds.
    
    Args:
        run_df: Dataframe containing each run (target and candidates) with at least its start and end time.
        
    Returns:
        run_df, but with added duration column
    """
    
    run_df['duration'] = (run_df['end_time'] - run_df['start_time']).dt.total_seconds()
    return run_df
