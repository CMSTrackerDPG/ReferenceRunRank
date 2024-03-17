import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from refrunrank.utilities import json_utils


class ranker:
    """
    Takes care of handling data used for ranking as well as the process of ranking iteself.
    """

    def __init__(self):
        pass


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
        grid = sns.pairplot(df, corner=True, plot_kws={"edgecolor": "none", "s": 1})
        grid.fig.set_dpi(150)
        for ax in grid.axes.flatten():
            if ax:
                ax.set_xlabel(ax.get_xlabel(), fontsize=10)
                ax.set_ylabel(ax.get_ylabel(), fontsize=10)
                ax.tick_params(axis="both", labelsize=8)
        plt.show()
    pca = PCA(n_components=1)
    pca.fit(df)
    pc1_loadings = pca.components_[0]
    pc1_df = pd.DataFrame(pc1_loadings, index=df.columns, columns=["PC1 Loadings"])
    pc1_df["weights"] = pc1_df["PC1 Loadings"] ** 2  # Already normalized

    return pc1_df["weights"].to_numpy()


def refrank(target_df, candidates_df, n_components=2):
    """
    Interphase to refrank_pca
    """
    rankings, wghts = refrank_pca(target_df, candidates_df, n_components=n_components)

    # Add a new 'rank' column with a range of values starting from 0
    rankings["rank"] = range(len(rankings))
    rankings.index.names = ["run"]

    # Set both 'rank' and 'run' as indices
    rankings = rankings.reset_index().set_index(["rank", "run"])

    # Construct a new dictionary mapping original column names to 'name (weight)' format
    new_column_names = {
        original: f"{original} ({round(wght, 4)})" for original, wght in wghts.items()
    }

    # Rename the columns using the new dictionary
    rankings.rename(columns=new_column_names, inplace=True)

    return rankings


def filter_runs(
    run_df,
    run_type,
    json_name=None,
    json_path="/eos/user/r/rcruzcan/SWAN_projects/RefRunRank/jsons/",
):
    """
    Removes NaN values, bad runs, keeping only runs of interest.
    """

    # Removing NaN values
    runclean_df = rm_nans(run_df, sort_by=["run_number"])

    # Filter by type.
    runclean_df = filter_by_type_run(runclean_df, run_type)

    # Filter out runs that are not in golden json
    if json_name is not None:
        goldenruns = map(
            int, list(json_utils.loadjson(os.path.join(json_path, json_name)).keys())
        )
        runclean_df = runclean_df[runclean_df["run_number"].isin(goldenruns)]

    return runclean_df


def filter_by_type_run(run_df, run_type):
    """
    Filter runs by type of run.
    """
    runfiltered_df = run_df[run_df["l1_hlt_mode"] == run_type].reset_index()

    if len(runfiltered_df) == 0:
        raise Exception("Error: No runs of the requested type were found.")

    return run_df


def filter_runs_runls(
    run_df,
    ls_df,
    json_name,
    run_type,
    json_path="/eos/user/r/rcruzcan/SWAN_projects/RefRunRank/jsons/",
    quality_threshold=0.75,
):
    """
    Removes NaN values, bad runs, keeping only runs of interest, and makes sure both LS and run dataframes have the same runs.

    """

    # Removing NaN values and setting indices
    run_df = rm_nans(run_df, sort_by=["run_number"])
    ls_df = rm_nans(ls_df, sort_by=["run_number", "lumisection_number"])

    # Filering out bad runs
    run_df, ls_df = filter_by_quality(
        ls_df, run_df, quality_threshold, json_path, json_name
    )

    # For now, we are only interested in collision2018 runs, so we discard the rest.
    run_df, ls_df = filter_by_type_runls(run_df, ls_df, run_type)

    run_df, ls_df = sync_dfs(run_df, ls_df)

    return run_df, ls_df


def rm_nans(df, sort_by=None, fillna_val={}):
    """
    Removes NaN values in the given dataframe and sorts by the given
    """
    df, _ = has_fill(df)
    df = df.convert_dtypes()
    if sort_by is not None:
        df.set_index(sort_by, inplace=True)
        df.sort_index(level=sort_by, inplace=True)
        # df.fillna(fillna_val, inplace=True)
        df.select_dtypes(include=["number"]).fillna(0, inplace=True)
        df.select_dtypes(include=["object"]).fillna("", inplace=True)
        df.reset_index(inplace=True)
    return df


def filter_by_quality(ls_df, run_df, quality_threshold, json_path, json_name):
    """
    Filters runs accoring to a quality threshold.
    """
    ls_df["is_good"] = json_utils.injson(
        np.array(ls_df["run_number"]),
        np.array(ls_df["lumisection_number"]),
        os.path.join(json_path, json_name),
    )
    quality_score = ls_df.groupby("run_number")["is_good"].mean()
    bad_runs = quality_score[quality_score < quality_threshold].index.tolist()

    runbad_df = run_df[run_df["run_number"].isin(bad_runs)]
    run_df = run_df[~run_df["run_number"].isin(bad_runs)]

    lsbad_df = ls_df[ls_df["run_number"].isin(bad_runs)]
    ls_df = ls_df[~ls_df["run_number"].isin(bad_runs)]

    return run_df, ls_df


def filter_by_type_runls(run_df, ls_df, run_type):
    """
    Filter runs and lumisections by type of run.
    """
    run_df = run_df[run_df["l1_hlt_mode"] == run_type]
    coll_runs = np.array(run_df["run_number"])
    ls_df = ls_df[ls_df["run_number"].isin(coll_runs)]
    return run_df, ls_df


def sync_dfs(run_df, ls_df):
    rundf_runs = np.array(run_df["run_number"])
    lsdf_runs = np.array(ls_df["run_number"])

    missing_runs = np.setxor1d(rundf_runs, lsdf_runs)

    run_df = run_df[~run_df["run_number"].isin(missing_runs)]
    ls_df = ls_df[~ls_df["run_number"].isin(missing_runs)]

    return run_df, ls_df


def lowcorr_highweight(
    features, corr_threshold, weight_threshold, plot_feats=False, plot_corrs=False
):
    """
    Function which extracts features with low correlation only and then applies filtering of the features based on the sum of the weights as determined by pca
    """
    #     ftr_names = list(features.columns)
    #     if "duration" in ftr_names:
    #         features = features.drop(columns=["duration"])
    #     if "last_lumisection_number" in ftr_names:
    #         features = features.drop(columns=["last_lumisection_number"])

    # Columns to exclude
    excluded_ftrs = ["duration", "last_lumisection_number", "fill_number"]

    included_ftrs = [col for col in features.columns if col not in excluded_ftrs]

    # Features which actually are used in PCA
    ftrs_PCA = features[included_ftrs]

    # Getting correlation and dropping highly correlated features
    corr_matrix = ftrs_PCA.corr()

    # Optional: Plot correlation matrix
    if plot_corrs:
        # Plotting correlation matrix
        plt.figure(figsize=(30, 30))
        # sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidth=0.5)
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", annot_kws={"size": 12})
        plt.title("Feature Correlation Matrix")
        plt.show()

    # Determining which features are too highly correlated
    to_drop = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                to_drop.add(corr_matrix.columns[j])

    # Dropping highly correlated features before getting weights
    features_lowcorr = ftrs_PCA.drop(columns=list(to_drop))

    # Getting weights and finding the set of most important features
    featureslowcorr_weights = get_weights(features_lowcorr, plot=plot_feats)

    ftrweightslowcorr_dict = {
        feature: round(weight, 4)
        for feature, weight in zip(
            features_lowcorr.columns.to_list(), featureslowcorr_weights
        )
    }
    ftrweightslowcorr_df = (
        pd.DataFrame(
            list(ftrweightslowcorr_dict.items()), columns=["Feature", "Weight"]
        )
        .sort_values("Weight", ascending=False)
        .reset_index(drop=True)
    )

    weight_sum = 0
    ftrs_to_use = []

    # Getting rid of unimportant features (as determined by threshold)
    for i, row in ftrweightslowcorr_df.iterrows():
        weight_sum += row["Weight"]
        ftrs_to_use.append(row["Feature"])
        if weight_sum > weight_threshold:
            break

    uncorr_imp_ftrs = ftrs_PCA[ftrs_to_use]

    # Output dataframe
    uncorr_imp_ftrs = ftrs_PCA[ftrs_to_use]

    if "last_lumisection_number" in features.columns:
        uncorr_imp_ftrs["last_lumisection_number"] = features["last_lumisection_number"]
    if "duration" in features.columns:
        uncorr_imp_ftrs["duration"] = features["duration"]
    if "fill_number" in features.columns:
        uncorr_imp_ftrs["fill_number"] = features["fill_number"]

    return uncorr_imp_ftrs


####


def has_fill(df):
    """
    Input: dataframe

    Returns: dataframe

    This function returns a dataframe where only entries that have a non-null "fill_number" attribute are kept.
    """
    df_nonNaN = df[df.fill_number.isna() == False].copy()
    df_NaN = df[df.fill_number.isna() == True].copy()

    return df_nonNaN, df_NaN


def addfillordernum(df):  # Test
    dfcopy = df.copy()
    dfcopy["fill_order_num"] = np.empty(len(dfcopy))
    dfcopy["num_runs_in_fill"] = np.empty(len(dfcopy))
    temp = dfcopy.groupby("fill_number")["run_number"].transform("count")
    x = 1
    for count, i in enumerate(temp):
        if x == i:
            dfcopy["fill_order_num"].iloc[count] = int(x)
            dfcopy["num_runs_in_fill"].iloc[count] = int(i)
            x = 1
            continue
        dfcopy["fill_order_num"].iloc[count] = int(x)
        dfcopy["num_runs_in_fill"].iloc[count] = int(i)
        x += 1

    return dfcopy


def add_loc_wrt_fill(df):
    import numpy as np

    dfcopy = df.copy()
    dfcopy["Fill location"] = np.empty(len(dfcopy))
    temp = dfcopy.groupby("fill_number")["run_number"].transform("count")
    x = 1
    for count, i in enumerate(temp):
        if x == i:
            dfcopy["Fill location"].iloc[count] = "({}/{})".format(x, i)
            x = 1
            continue
        dfcopy["Fill location"].iloc[count] = "({}/{})".format(x, i)
        x += 1

    return dfcopy


def makeDF(json):
    # if isinstance(json, dict):
    datadict = json["data"][0]["attributes"]
    keys = datadict.keys()

    datasetlist = []

    for i in range(len(json["data"])):
        values = json["data"][i]["attributes"].values()
        datasetlist.append(values)
    return pd.DataFrame(datasetlist, columns=keys)  # \
    # elif isinstance(json, list):
    # keys = json[0].keys()
    # for i in range(len(json)):...


def convert_check_addFillLoc(json):
    """
    Expects a json from with the attribute "fill_number" in the query

    """
    df = makeDF(json)
    # now filter out runs that don't  have fill number
    df = has_fill(df)
    # now add run location wrt fill
    DF_withloc = add_loc_wrt_fill(df)
    return DF_withloc


def get_collisions(rundf, lsdf):
    rundf_coll = rundf[rundf["l1_hlt_mode"].str.contains("collisions")]
    rundf_notcoll = rundf[~rundf["l1_hlt_mode"].str.contains("collisions")]

    lsdf_coll = lsdf[lsdf["run_number"].isin(rundf_coll["run_number"])]
    lsdf_notcoll = lsdf[lsdf["run_number"].isin(rundf_notcoll["run_number"])]

    return rundf_coll, lsdf_coll, rundf_notcoll, lsdf_notcoll


def get_runs_in_ls_df(lsdf):
    """
    Gets the list of run numbers in the lumi df
    """
    return lsdf["run_number"].unique()


def missing_runs(runsdf, lsdf, fromlumi=True):
    """
    Function takes both run and lumi df and find missing runs.
    Returns:  list of missing runs

    ----
    fromlumi :  defaults to True
                This will look for missing run numbers from the Lumi df
                if False : returns missing run numbers from the Run df
    """

    miss_runs = []
    runsINls = get_runs_in_ls_df(lsdf)
    if fromlumi:
        for i in runsdf["run_number"].values:
            if i not in runsINls:
                miss_runs.append(i)
    else:
        for i in runsINls:
            if i not in runsdf["run_number"].values:
                miss_runs.append(i)
    return miss_runs
