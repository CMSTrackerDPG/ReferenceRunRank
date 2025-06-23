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
        self.ftrnames = {"ftrs": [], "pca_ftrs": []}

    def setFeatures(self, ftrs: dict | str | None = None) -> dict:
        """Set features to be used for ranking."""
        if ftrs is None:
            self.ftrsdict = {"ftrs": {}, "pca_ftrs": {}}
            return self.ftrsdict
        elif isinstance(ftrs, str):
            import json

            try:
                with open(ftrs, "r") as f:
                    ftrs = json.load(f)
            except FileNotFoundError:
                raise FileNotFoundError(f"File {ftrs} not found.")

        self.ftrsdict = {
            "ftrs": ftrs.get("ftrs", {}),
            "pca_ftrs": ftrs.get("pca_ftrs", {}),
        }

        return self.ftrsdict

    def _unifyftrsdict(self, ftrsdict: dict) -> dict:
        """Unify the feature dictionary to ensure no duplicates and correct structure."""
        # Make sure at least one of "ftrs" or "pca_ftrs" exists
        if not ftrsdict.get("ftrs") and not ftrsdict.get("pca_ftrs"):
            raise ValueError(
                "At least one of 'ftrs' or 'pca_ftrs' must be present in the input."
            )

        merged_ftrs = {}

        # Handle "runs"
        runs_ftrs = ftrsdict.get("ftrs", {}).get(
            "runs", []
        )  # List of features for runs
        runs_pca_ftrs = ftrsdict.get("pca_ftrs", {}).get(
            "runs", []
        )  # List of PCA features for runs

        duplicate_runs_ftrs = set(runs_ftrs).intersection(runs_pca_ftrs)
        if duplicate_runs_ftrs:
            raise ValueError(
                f"Duplicate features found in 'runs': {duplicate_runs_ftrs}"
            )
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
                raise ValueError(
                    f"Duplicate features found in 'lumisections' for key '{key}': {duplicate_lumisections_ftrs}"
                )
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
        """Construct features dataframe that is used for ranking runs."""

        self.ftrsDF = pd.DataFrame(
            index=self.omsdata["runs"].index.to_list()
        )  # Find better approach to get list of runnbs
        self.ftrsDF.index.name = "runnb"
        feature_data = [self.ftrsDF]  # Collect dataframes to concatenate later

        for endpoint, ftrs in self._unifyftrsdict(self.ftrsdict).items():
            if endpoint == "runs":
                feature_data.append(self.omsdata["runs"][ftrs])
            elif endpoint == "lumisections":
                stats_df = (
                    self.omsdata["lumisections"]
                    .select_dtypes(include=[int, float])
                    .groupby("runnb")
                    .describe()
                )
                for stat_key, base_ftrs in ftrs.items():
                    # stat_key -> "mean", "std", "min", "max", etc.
                    # base_ftrs -> list of features to compute statistics on
                    feature_data.extend(
                        self._constructLSFeatures(stats_df, stat_key, base_ftrs)
                    )
            # Concatenate all dataframes at once
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")

        self.ftrsDF = pd.concat(feature_data, axis=1)
        self.ftrsDF.fillna(-999, inplace=True)
        self.ftrsDF = self.ftrsDF[
            ~self.ftrsDF.apply(lambda row: -999 in row.values, axis=1)
        ]
        self.ftrsDF_dropped = self.ftrsDF[
            self.ftrsDF.apply(lambda row: -999 in row.values, axis=1)
        ]
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
            raise AttributeError(
                "Features DataFrame (ftrsDF) has not been constructed. Call constructFeatures first."
            )

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
                for ftr_name, wght in zip(
                    pca.feature_names_in_, pca.components_[i] ** 2
                )
            }
            for i in range(n_components)
        ]

        return features_PC, wghts, pca

    def refrank(
        self,
        target_runnb=None,
        n_components=1,
        keep_ftrs=False,
        dist_metric="eucl",
    ):
        """Ranks runs according to their proximity to the target run."""
        if (
            target_runnb is None
        ):  # If no target provided, use the run with the highest run number
            print(
                "[WARNING] No target run number provided. Using the highest run number as target."
            )
            target_runnb = self.ftrsDF.index.max()

        # Making sure no future runs are included in the ranking
        features_ranking_all = self.ftrsDF[self.ftrsDF.index <= target_runnb]
        run_nums = pd.Series(features_ranking_all.index)

        scaler = StandardScaler()
        # Scale the features for ranking
        features_ranking_scaled = pd.DataFrame(
            scaler.fit_transform(features_ranking_all),
            columns=features_ranking_all.columns,
        )
        features_ranking_scaled = pd.concat(
            [run_nums, features_ranking_scaled], axis=1
        ).set_index("runnb")

        if self.ftrnames.get("pca_ftrs", None):
            features_ranking_PCA = features_ranking_scaled[self.ftrnames["pca_ftrs"]]
            features_PC, wghts, pca = self._pca_transform(
                features=features_ranking_PCA,
                target_runnb=target_runnb,
                n_components=n_components,
                scale=False,  # No scaling for PCA, as it is already scaled above
            )
            # Joining PCA features with the scaled non PCA features
            features_ranking = features_ranking_scaled[self.ftrnames["ftrs"]].join(
                features_PC, on="runnb"
            )
        else:
            features_ranking = features_ranking_scaled[self.ftrnames["ftrs"]]
            wghts = None
            pca = None

        if dist_metric == "eucl":
            dist = np.sqrt(
                ((features_ranking - features_ranking.loc[target_runnb]) ** 2).sum(
                    axis=1
                )
            )
        elif dist_metric == "manh":
            dist = (
                (features_ranking - features_ranking.loc[target_runnb])
                .abs()
                .sum(axis=1)
            )
        else:
            raise ValueError("Distance option not recognized. Must be 'eucl' or 'manh'")

        features_ranking["dist"] = dist.values

        if keep_ftrs:
            features_ranking[self.ftrnames["ftrs"]] = features_ranking_all[
                self.ftrnames["ftrs"]
            ]

        rankings = features_ranking.sort_values(by="dist", ascending=True).reset_index()

        return rankings, wghts, pca, scaler
