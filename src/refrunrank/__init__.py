# Credit: Gabriele Benelli

import json
import pandas as pd
from fnmatch import fnmatch


class CHRunData:
    """Creating a simple class to organize the Reference Runs information from the CertHelper API"""

    def __init__(self, JSONFilename):
        """Reading in the list of dictionaries in the "JSON" produced by the CertHelper API: e.g.  {"run_number": 306584, "run_reconstruction_type": "rereco", "reference_run_number": 305810, "reference_run_reconstruction_type": "express", "dataset": "/SingleTrack/Run2017G-17Nov2017-v1/DQMIO"}"""
        # self.JSONFilename=JSONFilename
        self.JSONFile = open(JSONFilename, "r")
        self.JSON = json.load(self.JSONFile)
        self.RunsDF = pd.DataFrame(self.JSON)
        self.RunsDF.dropna(inplace=True)
        self.filteredDF = None
        self.AllRunsByYear = {}

    def applyfilter(self, filters={}):
        if len(filters) == 0:
            print("Warning: No filter conditions given.")
            return self.RunsDF

        mask = pd.Series([True] * len(self.RunsDF), index=self.RunsDF.index)

        for key, value in filters.items():
            if isinstance(value, tuple) and key in [
                "run_number",
                "reference_run_number",
            ]:
                mask &= (self.RunsDF[key] >= value[0]) & (self.RunsDF[key] <= value[1])
            elif isinstance(value, list) and key in [
                "run_number",
                "reference_run_number",
            ]:
                mask &= self.RunsDF[key].isin(value)
            elif isinstance(value, (int, float)) and key in [
                "run_number",
                "reference_run_number",
            ]:
                mask &= self.RunsDF[key] == value
            elif isinstance(value, str) and key in [
                "run_reconstruction_type",
                "reference_run_reconstruction_type",
                "dataset",
            ]:
                mask &= self.RunsDF[key].apply(lambda x: fnmatch(x, value))

        # filteredDF = self.RunsDF[mask]
        self.filteredDF = self.RunsDF[mask]
        return self.filteredDF
