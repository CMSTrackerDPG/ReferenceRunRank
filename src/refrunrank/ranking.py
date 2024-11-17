import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster
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
        self.ftrsDF = pd.DataFrame(index=self.omsdata.getData().index.to_list()) # Find better approach to get list of runnbs
        self.ftrsDF.index.name = "run_number"
        feature_data = [self.ftrsDF]  # Collect dataframes to concatenate later
    
        for endpoint, ftrs in self._ftrsdict.items():
            if endpoint == "runs":
                feature_data.append(self.omsdata.getData("runs")[ftrs])
            elif endpoint == "lumisections":
                for stat_key, base_ftrs in ftrs.items():
                    lsftrs = self.constructLSFeatures(stat_key, base_ftrs)
                    feature_data.append(lsftrs)
    
        # Concatenate all dataframes at once
        self.ftrsDF = pd.concat(feature_data, axis=1)

    def constructLSFeatures(self, stat_key, base_ftrs):
        ls_df = self.omsdata.getData("lumisections").select_dtypes(include=[int, float])
        stats_df = pd.DataFrame()
        for runnb in ls_df.index.get_level_values(0).unique():
            newstats_df = ls_df.loc[runnb].describe()[base_ftrs].loc[stat_key]
            newstats_df["run_number"] = runnb
            stats_df = pd.concat([stats_df, pd.DataFrame(newstats_df).T.set_index("run_number")])
        stats_df = stats_df.add_suffix("_{}".format(stat_key))
        return stats_df

    def refrank_pca(
        self,
        target_runnb,
        n_components=1,
        keep_ftrs=True,
        selected_ftrs=None
    ):
        features = self.ftrsDF
        if selected_ftrs is not None:
            features = features[selected_ftrs]
 
        features = features[features.index <= target_runnb]
        run_nums = pd.Series(features.index)
        
        # Scaling for PCA
        scaler = StandardScaler()
        features = pd.DataFrame(
            scaler.fit_transform(features), columns=features.columns
        )

        # Finding PCs
        pca = PCA(n_components=n_components)
        pca.fit(features)

        # Constructing df w/ standardized features
        features_PC = pd.DataFrame(
            pca.transform(features),
            columns=["PC" + str(k + 1) for k in range(len(pca.components_))],
        )
        features_PC = pd.concat([run_nums, features_PC], axis=1).set_index("run_number")

        # Computing Eucledian distance in PCA sub-space
        dist = np.sqrt(((features_PC - features_PC.loc[target_runnb]) ** 2).sum(axis=1))
        features_PC["dist"] = dist.values
        features_PC = features_PC.sort_values(by="dist", ascending=True).reset_index()

        wghts = {
            ftr_name: wght
            for ftr_name, wght in zip(features.columns, pca.components_[0] ** 2)
        }

        return features_PC, wghts

    def hierarch_clust(self, target_runnb, n_components=1, corr_thrshld=0.7, rtrn_stats=False):
        _, wghts = self.refrank_pca(target_runnb, n_components=n_components)
        wghts_df = pd.DataFrame(list(wghts.items()), columns=["Feature", "Weight"])
        wghts_df = wghts_df.sort_values(by="Weight", ascending=False).reset_index(drop=True)
        
        corr_mtrx = self.ftrsDF.corr()
        corr_mtrx = corr_mtrx.fillna(0)
        corr_dist = 1-corr_mtrx.abs()
        
        dist_condensed = corr_dist.values[np.triu_indices_from(corr_dist, k=1)]
        linkage_mtrx = linkage(dist_condensed, method="complete")
        clusters = fcluster(linkage_mtrx, t=corr_thrshld, criterion="distance")
        
        clustered_ftrs = {}
        for idx, cluster_id in enumerate(clusters):
            feature = corr_mtrx.columns[idx]
            if cluster_id not in clustered_ftrs:
                clustered_ftrs[cluster_id] = [feature]
            else:
                clustered_ftrs[cluster_id].append(feature)
        selected_ftrs = []
        
        for cluster, features in clustered_ftrs.items():
            if len(features) == 1:
                selected_ftrs.append(features[0])
            else:
                top_ftr_idx = wghts_df[wghts_df["Feature"].isin(features)]["Weight"].idxmax()
                top_ftr = wghts_df.iloc[top_ftr_idx]["Feature"]
                selected_ftrs.append(top_ftr)

        if rtrn_stats:
            order = leaves_list(linkage_mtrx)
            ordered_corr_mtrx = corr_mtrx.iloc[order, order]
            return selected_ftrs, ordered_corr_mtrx, linkage_mtrx
        return selected_ftrs

    def refrank_pca_hierarch(self, target_runnb, n_components=1, keep_ftrs=True, corr_thrshld=0.7):
        selected_ftrs = self.hierarch_clust(target_runnb, n_components=n_components, corr_thrshld=corr_thrshld)
        rslts, wghts = self.refrank_pca(target_runnb, n_components=n_components, keep_ftrs=keep_ftrs, selected_ftrs=selected_ftrs)
        return (rslts, wghts)

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