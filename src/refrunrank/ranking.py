import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
from refrunrank.utils import data_utils

class RunRanker:
    """
    Class which takes as input applied ranking algorithm.
    """

    def __init__(self, omsdata, ftrs=[], weight_thrshld=1, corr_thrshld=1):
        self._setOMSData(omsdata)

    def _setOMSData(self, omsdata):
        self.omsdata = omsdata

    def setFeatures(self, ftrsdict):
        self._ftrsdict = ftrsdict

    def getFeatures(self):
        return self._ftrsdict

    def constructFeatures(self):
        self.ftrsDF = pd.DataFrame(index=self.omsdata.getRunnbs())
        self.ftrsDF.index.name = "run_number"
        for endpoint, ftrs in self._ftrsdict.items():
            if endpoint == "runs":
                self.ftrsDF = pd.concat([self.ftrsDF, self.omsdata.getData("runs")[ftrs]], axis=1)
            if endpoint == "lumisections":
                for stat_key, base_ftrs in ftrs.items():
                    lsftrs = self.constructLSFeatures(stat_key, base_ftrs)
                    self.ftrsDF = pd.concat([self.ftrsDF, lsftrs], axis=1)

    def constructLSFeatures(self, stat_key, base_ftrs):
        ls_df = self.omsdata.getData("lumisections").select_dtypes(include=[int, float])
        stats_df = pd.DataFrame()
        for runnb in ls_df.index.get_level_values(0).unique():
            newstats_df = ls_df.loc[runnb].describe()[base_ftrs].loc[stat_key]
            newstats_df["run_number"] = runnb
            stats_df = pd.concat([stats_df, pd.DataFrame(newstats_df).T.set_index("run_number")])
        stats_df = stats_df.add_suffix("_{}".format(stat_key))
        return stats_df
        
                
    def refrank(self, target, n_components=2):
        """
        Interphase to refrank_pca
        """
        if target not in self.ftrsDF.reset_index()["run_number"].values:
            raise ValueError("Data of target run not loaded.")

        # selected_ftrs = [col for col, include in self.rankftrs.items() if include]
        target_df = pd.DataFrame(self.ftrsDF.loc[target]).T
        candidates_df = self.rundf[self.ftrsDF.index < target]

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
        target_runnb,
        n_components=None,
        keep_ftrs=True,
    ):
 
        features = self.ftrsDF[self.ftrsDF.index <= target_runnb]
        run_nums = pd.Series(features.index)
        
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
        features_PC = pd.concat([run_nums, features_PC], axis=1).set_index("run_number")

        # Computing Eucledian distance in PCA space
        dist = np.sqrt(((features_PC - features_PC.loc[target_runnb]) ** 2).sum(axis=1))
        features_PC["dist"] = dist.values
        features_PC = features_PC.sort_values(by="dist", ascending=True).reset_index()

        wghts = {
            ftr_name: wght
            for ftr_name, wght in zip(features.columns, pca.components_[0] ** 2)
        }

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