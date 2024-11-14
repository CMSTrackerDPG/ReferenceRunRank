from refrunrank.utils.data_utils import loadJSONasDF, loadFromWeb
from fnmatch import fnmatch
import pandas as pd

class CHRunData:
    """
    Class to organize the Reference Runs information from the CertHelper API
    Credit for original JSON implementation: Gabriele Benelli
    """

    def __init__(self, JSONFilePath, goldenJSONFilePath=None, filtergolden=True):
        """
        Reading in the list of dictionaries in the "JSON" produced by the CertHelper API: e.g.  {"run_number": 306584, "run_reconstruction_type": "rereco", "reference_run_number": 305810, "reference_run_reconstruction_type": "express", "dataset": "/SingleTrack/Run2017G-17Nov2017-v1/DQMIO"}
        """
        self.RunsDF = loadJSONasDF(JSONFilePath)
        self.RunsDF.dropna(inplace=True)
        self._setGolden(goldenJSONFilePath)
        self.RunsDF.sort_values("run_number", inplace=True)

    def _setGolden(self, goldenJSONFilePath=None):
        if goldenJSONFilePath is None:
            return
        self.goldenDF = loadJSONasDF(goldenJSONFilePath)
        self.goldenDF = self.goldenDF.rename(
            {0: "run_number", 1: "good_lss"}, axis=1
        )
        self.goldenDF = self.goldenDF.astype({"run_number": int})

        # Put golden info in RunsDF
        self.RunsDF = self.RunsDF.merge(self.goldenDF, on="run_number", how="left")
        self.RunsDF["good_lss"] = self.RunsDF["good_lss"].where(self.RunsDF["good_lss"].notna(), None)

    def getGoodRuns(self):
        return self.RunsDF[self.RunsDF["good_lss"].notnull()]

    def getRuns(self, exclude_bad=True):
        if exclude_bad:
            return getGoodRuns()
        else:
            return self.RunsDF

    def getRun(self, runnb):
        return self.RunsDF[self.RunsDF["run_number"] == runnb]

    def applyFilter(self, exclude_bad=True, filters={}):
        if exclude_bad:
            RunsDF = self.getGoodRuns()
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
        return RunsDF[mask]

    def getruns(self, run, colfilters=None):
        """
        Simpler function to get particular runs and their features. Intended mostly for testing
        """
        CHftrs = [
            "run_number",
            "run_reconstruction_type",
            "reference_run_type",
            "reference_run_reconstruction_type",
            "dataset",
        ]
        try:
            runs = self.RunsDF[self.RunsDF["run_number"] == run]
            if colfilters is None:
                return runs
            else:
                if isinstance(colfilters, list):
                    badftrs = []
                    for colfilter in colfilters:
                        if colfilter not in CHftrs:
                            badftrs.append(colfilter)
                            print(
                                "WARNING: {} not a valid CH feature. Skipping.".format(
                                    colfilter
                                )
                            )
                    return runs[list(set(colfilters) - set(badftrs))]
                else:
                    raise Exception("colfilters must be of type list")
        except:
            raise Exception("Run is not available.")