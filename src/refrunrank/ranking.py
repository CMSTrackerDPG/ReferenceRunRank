import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster

class RunRanker:

    def __init__(self, omsdata, ftrs=None):
        self.omsdata = omsdata
        self.setFeatures(ftrs)
        self.ftrsDF = None
        self.ftrnames = {}

    def setFeatures(self, ftrs: dict | str | None = None) -> None:
        """Set features to be used for ranking."""
        if isinstance(ftrs, str):
            import json
            try:
                with open(ftrs, 'r') as f:
                    ftrs = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {ftrs} not found.")
            ftrs = {k: v for k, v in ftrs.items() if k in ["ftrs", "pca_ftrs"]}

        if len(ftrs) == 0:
            ftrs = None

        self.ftrsdict = ftrs
        return self.ftrsdict

    def _unifyftrsdict(self, ftrsdict: dict) -> dict:
        """Unify the feature dictionary to ensure no duplicates and correct structure."""
        # Make sure at least one of "ftrs" or "pca_ftrs" exists
        if not ftrsdict.get("ftrs") and not ftrsdict.get("pca_ftrs"):
            raise ValueError("At least one of 'ftrs' or 'pca_ftrs' must be present in the input.")
        
        merged_ftrs = {}

        # Handle "runs"
        runs_ftrs = ftrsdict.get("ftrs", {}).get("runs", []) # List of features for runs
        runs_pca_ftrs = ftrsdict.get("pca_ftrs", {}).get("runs", []) # List of PCA features for runs

        duplicate_runs_ftrs = set(runs_ftrs).intersection(runs_pca_ftrs)
        if duplicate_runs_ftrs:
            raise ValueError(f"Duplicate features found in 'runs': {duplicate_runs_ftrs}")
        merged_ftrs["runs"] = runs_ftrs + runs_pca_ftrs

        # Handle "lumisections"
        lumisections_ftrs = ftrsdict.get("ftrs", {}).get("lumisections", {})
        lumisections_pca_ftrs = ftrsdict.get("pca_ftrs", {}).get("lumisections", {})
        lumisections = {}

        for key in set(lumisections_ftrs.keys()).union(lumisections_pca_ftrs.keys()):
            list_ftrs = lumisections_ftrs.get(key, [])
            list_pca_ftrs = lumisections_pca_ftrs.get(key, [])
            duplicate_lumisections_ftrs = set(list_ftrs).intersection(list_pca_ftrs)
            if duplicate_lumisections_ftrs:
                raise ValueError(f"Duplicate features found in 'lumisections' for key '{key}': {duplicate_lumisections_ftrs}")
            lumisections[key] = list_ftrs + list_pca_ftrs

        merged_ftrs["lumisections"] = lumisections

        return merged_ftrs
    
    def _constructLSFeatures(self, stats_df, stat_key, base_ftrs):
        """Construct summary statistics for LS-level features."""

        statftrs_cols = []
        for ftr in base_ftrs:
            statftrs_cols.append(
                pd.DataFrame(stats_df[ftr][stat_key]).add_prefix(f"{ftr}_")
            )
        return statftrs_cols
    
    def constructFeatures(self):
        """Costruct features dataframe that is used for ranking runs."""

        self.ftrsDF = pd.DataFrame(index=self.omsdata["runs"].index.to_list()) # Find better approach to get list of runnbs
        self.ftrsDF.index.name = "runnb"
        feature_data = [self.ftrsDF]  # Collect dataframes to concatenate later
    
        for endpoint, ftrs in self._unifyftrsdict(self.ftrsdict).items():
            if endpoint == "runs":
                feature_data.append(self.omsdata["runs"][ftrs])
            elif endpoint == "lumisections":
                stats_df = self.omsdata["lumisections"].select_dtypes(include=[int, float]).groupby("runnb").describe()
                for stat_key, base_ftrs in ftrs.items():
                    # stat_key -> "mean", "std", "min", "max", etc.
                    # base_ftrs -> list of features to compute statistics on
                    feature_data.extend(self._constructLSFeatures(stats_df, stat_key, base_ftrs))
            # Concatenate all dataframes at once
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")

        self.ftrsDF = pd.concat(feature_data, axis=1)
        self.ftrsDF.fillna(-999, inplace=True)
        self.ftrsDF = self.ftrsDF[~self.ftrsDF.apply(lambda row: -999 in row.values, axis=1)]
        self.ftrsDF_dropped = self.ftrsDF[self.ftrsDF.apply(lambda row: -999 in row.values, axis=1)]
        self.ftrsDF.sort_index(inplace=True)

        self.ftrnames = {"ftrs": [], "pca_ftrs": []}
        for ftr_type, ftr_dict in self.ftrsdict.items():
            for ftr_category, ftr in ftr_dict.items():
                if ftr_category == "runs":
                    self.ftrnames[ftr_type].extend(ftr)
                elif ftr_category == "lumisections":
                    for stat_key, base_ftrs in ftr.items():
                        for base_ftr in base_ftrs:
                            self.ftrnames[ftr_type].append(f"{base_ftr}_{stat_key}")

        return self.ftrsDF

    def _filterRuns(self, runnbs=None):
        """Returns feature dataframe of specified run numbers."""
        if runnbs is None:
            return self.ftrsDF
        return self.ftrsDF[self.ftrsDF.index.isin(runnbs)]

    def _pca_transform(
        self, 
        features: pd.DataFrame | None = None, 
        n_components: int = 1, 
        target_runnb: int | None = None, 
        use_all_ftrs: bool = False,
        runnbs: list[int] | None = None, 
        scale: bool = True,
    ):
        """Returns PCA-transformed feature DataFrame and weights, excluding any ranking logic."""
        
        if not hasattr(self, "ftrsDF") or self.ftrsDF is None:
            raise AttributeError("Features DataFrame (ftrsDF) has not been constructed. Call constructFeatures first.")

        if features is None:
            # If no features are provided, use the full DataFrame
            features = self._filterRuns(runnbs)

        if self.ftrnames.get("pca_ftrs", None) is not None and not use_all_ftrs:
            # If pca_ftrs are specified, filter the features DataFrame to include only those
            features = features[self.ftrnames["pca_ftrs"]]
            
        if target_runnb is not None:
            # If a target run number is specified, filter the features DataFrame 
            # to include only runs up to that number
            features = features[features.index <= target_runnb]

        if features.empty:
            raise ValueError("No runs available for PCA after filtering.")

        run_nums = pd.Series(features.index)

        if scale:
            scaler = StandardScaler()
            features_scaled = pd.DataFrame(
                scaler.fit_transform(features), columns=features.columns
            )
        else:
            # No scaling, just for testing purposes
            features_scaled = features.copy()

        pca = PCA(n_components=n_components)
        pca.fit(features_scaled)

        features_PC = pd.DataFrame(
            pca.transform(features_scaled),
            columns=["PC" + str(k + 1) for k in range(len(pca.components_))],
        )
        features_PC = pd.concat([run_nums, features_PC], axis=1).set_index("runnb")

        wghts = [
            {
                ftr_name: float(wght)
                for ftr_name, wght in zip(pca.feature_names_in_, pca.components_[i] ** 2)
            } for i in range(n_components)
        ]

        return features_PC, wghts, pca

    def refrank(
        self,
        target_runnb = None,
        n_components=1,
        keep_ftrs=False,
        dist_metric="eucl",
    ):
        """Ranks runs according to their proximity to the target run."""
        if target_runnb is None: # If no target provided, use the run with the highest run number
            print("WARNING: No target run number provided. Using the highest run number as target.")
            target_runnb = self.ftrsDF.index.max()

        # Making sure no future runs are included in the ranking
        features_ranking_all = self.ftrsDF[self.ftrsDF.index <= target_runnb] 
        run_nums = pd.Series(features_ranking_all.index)
        

        scaler = StandardScaler()
        # Scale the features for ranking
        features_ranking_scaled = pd.DataFrame(scaler.fit_transform(features_ranking_all), columns=features_ranking_all.columns)
        features_ranking_scaled = pd.concat([run_nums, features_ranking_scaled], axis=1).set_index("runnb")
        
        features_ranking_PCA = features_ranking_scaled[self.ftrnames["pca_ftrs"]]
        features_PC, wghts, pca = self._pca_transform(
            features=features_ranking_PCA,
            target_runnb=target_runnb,
            n_components=n_components,
            scale=False,  # No scaling for PCA, as it is already scaled above
        )

        # Joining PCA features with the scaled non PCA features
        features_ranking = features_ranking_scaled[self.ftrnames["ftrs"]].join(features_PC, on="runnb")

        if dist_metric == "eucl":
            dist = np.sqrt(((features_ranking - features_ranking.loc[target_runnb]) ** 2).sum(axis=1))
        elif dist_metric == "manh":
            dist = (features_ranking - features_ranking.loc[target_runnb]).abs().sum(axis=1)
        else:
            raise ValueError("Distance option not recognized. Must be 'eucl' or 'manh'")

        features_ranking["dist"] = dist.values

        if keep_ftrs:
            features_ranking[self.ftrnames["ftrs"]] = features_ranking_all[self.ftrnames["ftrs"]]

        rankings = features_ranking.sort_values(by="dist", ascending=True).reset_index()


        return rankings, wghts, pca, scaler

    def hierarch_clust(self, target_runnb, n_components=1, corr_thrshld=0.7, runnbs=None, rtrn_stats=False, dist_metric="eucl"):
        """Selected features by their correlation and PC1 assigned weight"""
        filtered_ftrsDF = self.filterRuns(runnbs)
    
        _, wghts = self.refrank(target_runnb, n_components=n_components, runnbs=runnbs, dist_metric=dist_metric)
        wghts_df = pd.DataFrame(list(wghts.items()), columns=["Feature", "Weight"])
        wghts_df = wghts_df.sort_values(by="Weight", ascending=False).reset_index(drop=True)
        
        # Clustering
        corr_mtrx = filtered_ftrsDF.corr()
        corr_mtrx = corr_mtrx.fillna(0)
        corr_dist = 1 - corr_mtrx.abs()
        
        dist_condensed = corr_dist.values[np.triu_indices_from(corr_dist, k=1)]
        linkage_mtrx = linkage(dist_condensed, method="complete")
        clusters = fcluster(linkage_mtrx, t=corr_thrshld, criterion="distance")

        # Group by cluster
        clustered_ftrs = {}
        for idx, cluster_id in enumerate(clusters):
            feature = corr_mtrx.columns[idx]
            if cluster_id not in clustered_ftrs:
                clustered_ftrs[cluster_id] = [feature]
            else:
                clustered_ftrs[cluster_id].append(feature)
    
        # Ftr selection
        selected_ftrs = []
        for cluster, features in clustered_ftrs.items():
            top_wghts = wghts_df[wghts_df["Feature"].isin(features)]
            selected_ftrs.append(
                top_wghts.loc[top_wghts["Weight"].idxmax(), "Feature"]   
            )
    
        if rtrn_stats:
            from scipy.cluster.hierarchy import leaves_list
            order = leaves_list(linkage_mtrx)
            ordered_corr_mtrx = corr_mtrx.iloc[order, order]
            return selected_ftrs, ordered_corr_mtrx, linkage_mtrx
        
        return selected_ftrs

    
    def refrank_pca_hierarch(self, target_runnb, n_components=1, keep_ftrs=True, corr_thrshld=0.7, runnbs=None, dist_metric="eucl"):
        """
        Combines PCA ranking with hierarchical clustering to avoid using highly correlated features.
        """
        selected_ftrs = self.hierarch_clust(target_runnb, n_components=n_components, corr_thrshld=corr_thrshld, runnbs=runnbs, dist_metric=dist_metric)
        rslts, wghts = self.refrank(target_runnb, n_components=n_components, keep_ftrs=keep_ftrs, selected_ftrs=selected_ftrs, runnbs=runnbs, dist_metric=dist_metric)
        return rslts, wghts

    
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