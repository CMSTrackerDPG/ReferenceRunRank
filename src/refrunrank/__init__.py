import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from fnmatch import fnmatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from refrunrank.omsftrs import rankingftrs, filteringftrs
from typing import Union


class RunRanker:
    """
    Class which takes as input applied ranking algorithm.
    """

    def __init__(
        self, rundf: pd.DataFrame, ftrs=[], weight_thrshld=1, corr_thrshld=1
    ) -> None:
        # self.saved_views = []

        # Feature setup
        self.filteringftrs = filteringftrs
        self.rankftrnames = rankingftrs
        self.rankftrs = {ftr: False for ftr in self.rankftrnames}
        self.selectomsftrs(ftrs)

        # TODO: Auto feature selection given weights and corr threshold
        self.weight_thrshld = weight_thrshld  #
        self.corr_thrshld = corr_thrshld  #

        # Cleaning up and formatting input dataframe
        self.rundf = rundf
        self._formatdf()
        self._cleandf()

        # TODO
        self.viewdf = None
        self.latest_rankings = []

    def _formatdf(self) -> None:
        """
        Applies all neccesary formatting to the input dataframe.

        """
        self.rundf["run"] = self.rundf["run_number"]
        self.rundf.set_index("run", inplace=True)

    def _cleandf(self) -> None:
        """
        Removes NaN values in the given dataframe
        """
        self.rundf = self.rundf[~self.rundf["fill_number"].isna()]
        self.rundf = self.rundf.convert_dtypes()
        self.rundf.select_dtypes(include=["number"]).fillna(0, inplace=True)
        self.rundf.select_dtypes(include=["object"]).fillna("", inplace=True)
        self.rundf.drop("index", axis=1, inplace=True)

    def selectomsftrs(self, selected_ftrs: list) -> None:
        """
        Modifies selected features dictionary to reflect user's choice of ranking features
        """
        if True in self.rankftrs.values():
            print("Warning: Removing previously selected features")
            self._resetftrselection()

        for ftr in selected_ftrs:
            if ftr in self.rankftrnames:
                self.rankftrs[ftr] = True
            elif ftr in self.filteringftrs:
                print("Warning: {} is a filtering feature. Skipped.".format(ftr))
            else:
                raise Exception("{} is not a valid feature name".format(ftr))

    def _resetftrselection(self) -> None:
        """
        Resets the feature selection dictionary
        """
        for ftr in self.rankftrs:
            self.rankftrs[ftr] = False

    def refrank(self, target: int, n_components=2) -> pd.DataFrame:
        """
        Interphase to refrank_pca
        """
        if target not in self.rundf.reset_index()["run"].values:
            print("Warning: Target run not in given dataframe. Aborting.")
            return None

        selected_ftrs = [col for col, include in self.rankftrs.items() if include]
        target_df = pd.DataFrame(self.rundf.loc[target][selected_ftrs]).T
        candidates_df = self.rundf[self.rundf.index < target][selected_ftrs]

        rankings, wghts = self.refrank_pca(
            target_df, candidates_df, n_components=n_components
        )

        # Add a new 'rank' column with a range of values starting from 0
        rankings["rank"] = range(len(rankings))
        rankings.index.names = ["run"]

        # Set both 'rank' and 'run' as indices
        rankings = rankings.reset_index().set_index(["rank", "run"])

        # Construct a new dictionary mapping original column names to 'name (weight)' format
        new_column_names = {
            original: f"{original} ({round(wght, 4)})"
            for original, wght in wghts.items()
        }

        # Rename the columns using the new dictionary
        rankings.rename(columns=new_column_names, inplace=True)

        self.latest_rankings.append(rankings)

        return rankings

    def refrank_pca(
        self,
        targ_ftrs: pd.DataFrame,
        cands_ftrs: pd.DataFrame,
        n_components=None,
        keep_ftrs=True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Takes in a dataframe of features and ranks each point in feature space by their closseness to the target point. Distance measured in the n-th dimensional PCA sub-space using Eucledian distance. Assumptions:
            - Last row in `features` is the target dataframe
            - Index column is `run`, regardless if the dataframe has `run_number` as a feature
            - Dataframe has `last_lumisection_number` as one of the columns, but this is not used for ranking, just filtering.
            - The dataframe is assumed to already have all of the runs which will be used in comparison, thus removing the neccessity to have a comparison number parameter.

        Args:
            features: Dataframe containing runs along with the features to be considered during ranking. Must include an "run" index column which is used to distinguish each datapoint and `last_lumisection_number` column to filter by run length
            n_components: Number of dimensions of the PCA sub-space (i.e. number of PCA axis used)

        Returns:
            Input dataframe, ordered by rank, with an added column(s) showing the PC component(s).

        """

        target = targ_ftrs.index[0]
        features = pd.concat([targ_ftrs, cands_ftrs])
        features.index.name = "run"
        run_nums = features.reset_index()["run"]

        if keep_ftrs:
            ftrs_backup = features.copy(deep=True)

        # Scaling for PCA
        scaler = StandardScaler()
        features = pd.DataFrame(
            scaler.fit_transform(features), columns=features.columns
        )

        # If n_components not, specified, set it to max.
        if n_components is None:
            n_components = len(features.columns)

        # Finding PCs
        pca = PCA(n_components=n_components)
        pca.fit(features)

        # Constructing dataframe with standardized features
        features_PC = pd.DataFrame(
            pca.transform(features),
            columns=["PC" + str(k + 1) for k in range(len(pca.components_))],
        )
        features_PC = pd.concat([run_nums, features_PC], axis=1).set_index("run")

        # Computing Eucledian distance using coordinates in PCA sub-space and sorting by distance to target run
        dist = np.sqrt(((features_PC - features_PC.loc[target]) ** 2).sum(axis=1))
        features_PC["dist"] = dist.values
        features_PC = features_PC.sort_values(by="dist", ascending=True).reset_index()
        features_PC.set_index("run", inplace=True)

        wghts = {
            ftr_name: wght
            for ftr_name, wght in zip(features.columns, pca.components_[0] ** 2)
        }

        if keep_ftrs:
            features_PC = pd.concat([features_PC, ftrs_backup], axis=1)

        return features_PC, wghts

    def get_weights(self, df: pd.DataFrame, standardize=True, plot=False) -> np.ndarray:
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

    def lowcorr_highweight(
        self,
        features,
        corr_threshold,
        weight_threshold,
        plot_feats=False,
        plot_corrs=False,
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
        featureslowcorr_weights = self.get_weights(features_lowcorr, plot=plot_feats)

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
            uncorr_imp_ftrs["last_lumisection_number"] = features[
                "last_lumisection_number"
            ]
        if "duration" in features.columns:
            uncorr_imp_ftrs["duration"] = features["duration"]
        if "fill_number" in features.columns:
            uncorr_imp_ftrs["fill_number"] = features["fill_number"]

        return uncorr_imp_ftrs


class CHRunData:
    """
    Class to organize the Reference Runs information from the CertHelper API
    Credit for original JSON implementation: Gabriele Benelli
    """

    def __init__(self, JSONFilePath, goldenJSONFilePath=None, filtergolden=True):
        """Reading in the list of dictionaries in the "JSON" produced by the CertHelper API: e.g.  {"run_number": 306584, "run_reconstruction_type": "rereco", "reference_run_number": 305810, "reference_run_reconstruction_type": "express", "dataset": "/SingleTrack/Run2017G-17Nov2017-v1/DQMIO"}"""
        self.RunsDF = self._loadJSONasDF(JSONFilePath)
        self.RunsDF.dropna(inplace=True)

        self.filteredDF = None
        self.AllRunsByYear = {}

        self.goldenDF = None
        self.goldenRunsDF = None
        if goldenJSONFilePath is not None:
            self.goldenDF = self._loadJSONasDF(goldenJSONFilePath)
            self.goldenDF = self.goldenDF.T.reset_index().rename(
                {"index": "run", 0: "LSs"}, axis=1
            )
            self.goldenDF = self.goldenDF.astype({"run": int})
            self.filtergolden()

    def _loadJSONasDF(self, JSONFilePath):
        ### load the content of a json file into a python object
        # input arguments:
        # - jsonfile: the name (or full path if needed) to the json file to be read
        # output:
        # - an dict object as specified in the note below
        # note: the json file is supposed to contain an object like this example:
        #       { "294927": [ [ 55,85 ], [ 95,105] ], "294928": [ [1,33 ] ] }
        #       although no explicit checking is done in this function,
        #       objects that don't have this structure will probably lead to errors further in the code
        if not os.path.exists(JSONFilePath):
            raise Exception(
                "ERROR in json_utils.py / loadjson: requested json file {} does not seem to exist...".format(
                    JSONFilePath
                )
            )
        with open(JSONFilePath) as f:
            JSONdict = json.load(f)
        return pd.DataFrame(JSONdict).convert_dtypes()

    def filtergolden(self):
        if self.goldenDF is None:
            raise Exception("No golden JSON loaded.")
        self.goldenRunsDF = self.RunsDF[
            self.RunsDF["run_number"].isin(self.goldenDF["run"])
        ]

        return self.goldenRunsDF

    def applyfilter(self, filters={}):
        if self.goldenRunsDF is not None:
            try:
                RunsDF = self.goldenRunsDF
            except:
                raise Exception("Could not use goldenRunsDF to apply filters")
        else:
            RunsDF = self.RunsDF

        if len(filters) == 0:
            print("Warning: No filter conditions given.")
            return RunsDF

        mask = pd.Series([True] * len(RunsDF), index=RunsDF.index)

        for key, value in filters.items():
            if isinstance(value, tuple) and key in [
                "run_number",
                "reference_run_number",
            ]:
                mask &= (RunsDF[key] >= value[0]) & (RunsDF[key] <= value[1])
            elif isinstance(value, list) and key in [
                "run_number",
                "reference_run_number",
            ]:
                mask &= RunsDF[key].isin(value)
            elif isinstance(value, (int, float)) and key in [
                "run_number",
                "reference_run_number",
            ]:
                mask &= RunsDF[key] == value
            elif isinstance(value, str) and key in [
                "run_reconstruction_type",
                "reference_run_reconstruction_type",
                "dataset",
            ]:
                mask &= RunsDF[key].apply(lambda x: fnmatch(x, value))

        # filteredDF = self.RunsDF[mask]
        self.filteredDF = RunsDF[mask]
        return self.filteredDF
