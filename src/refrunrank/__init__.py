# Credit: Gabriele Benelli

import json
import pandas as pd
# CertHelperAPIJSONFilename="AllRunsRefRuns_Mar5_2024.json"
# RefRunsJSONFile=open(CertHelperAPIJSONFilename, 'r')
# RefRunsJSON=json.load(RefRunsJSONFile)
class CHRunData:
    """Creating a simple class to organize the Reference Runs information from the CertHelper API"""

    def __init__(self, JSONFilename):
        """Reading in the list of dictionaries in the "JSON" produced by the CertHelper API: e.g.  {"run_number": 306584, "run_reconstruction_type": "rereco", "reference_run_number": 305810, "reference_run_reconstruction_type": "express", "dataset": "/SingleTrack/Run2017G-17Nov2017-v1/DQMIO"}"""
        # self.JSONFilename=JSONFilename
        self.JSONFile = open(JSONFilename, "r")
        self.JSON = json.load(self.JSONFile)
        self.RunsDF = pd.DataFrame(self.JSON)
        self.RunsDF.dropna(inplace=True)
        self.AllRunsByYear = {}

    def getListOfRuns(self, type="all"):
        """Processing the data in the "JSON" to produce the interesting lists of runs"""
        if type == "all":
            self.AllRuns = []
            for entry in self.JSON:
                runNumber = entry["run_number"]
                self.AllRuns.append(
                    runNumber
                )  # Add all entries in this list, do more (cleaning/filtering) operations later
                # if runNumber not in self.AllRuns:
                #    self.AllRuns.append(runNumber)
            return self.AllRuns
        if type in ["2016", "2017", "2018", "2022", "2023", "2024"]:
            self.AllRunsByYear[type] = []
            for entry in self.JSON:
                # print(type)
                # print(entry['run_number'])
                # print(entry['dataset'])
                # Found that we have some dataset set to None, breaking the code, so catching the issue:
                if isinstance(entry["dataset"], str):
                    if type in entry["dataset"]:
                        runNumber = entry["run_number"]
                        self.AllRunsByYear[type].append(
                            runNumber
                        )  # Add all entries in this list, do more (cleaning/filtering) operations later
                else:
                    print(
                        "The following entry does not have a dataset string in the input JSON!"
                    )
                    print(entry)
            return self.AllRunsByYear[type]

    def getInfoForRun(self, runNumber):
        """Quick function to dump information in the JSON about a given run"""
        self.getListOfRuns()
        if int(runNumber) not in self.AllRuns:
            print(
                "Sorry run %s was not found in the currently loaded JSON file %s"
                % (runNumber, self.JSONFilename)
            )
            # return self.AllRuns
            return
        else:
            entries = []
            for entry in self.JSON:
                if entry["run_number"] == int(runNumber):
                    # print(entry)
                    entries.append(entry)
            return entries
