import pandas as pd
from cmsdials.filters import OMSFilter, OMSPage
from concurrent.futures import ThreadPoolExecutor, TimeoutError

class OMSData:
    def __init__(self, dials):
        self.dials = dials
        self.runfilters = [] # List of filters for runs
        self.filters = [] # List of other types of filters (e.g. max num lss)
        self.endpoints = [
            "runs",
            "runkeys",
            "l1configurationkey", # Not working
            "l1algorithmtriggers",
            "hltconfigdata",
            "deadtime", # 
            "daqreadouts",
            "fill", #
            "l1triggerrate", # Not working
            "lumisections"
        ]
        self._resetDataDict()

    def __getitem__(self, endpoint):
        return self._data[endpoint]

    def _resetDataDict(self, endpoint="all"):
        if endpoint == "all":
            self._data = {endpoint: None for endpoint in self.endpoints}
        else:
            self._data[endpoint] = None

    def _resetFilters(self):
        self.filters = []
        
    def _resetRunFilters(self):
        self.runfilters = []
        
    def getEndpoints(self):
        return self.endpoints

    def getRunFilters(self):
        return self.runfilters

    def getRunnbs(self):
        runnbs = []
        for runFilter in self.runfilters:
            runnbs.append(runFilter.value)
        return runnbs
        
    def getFilters(self):
        return self.filters

    def getData(self, endpoint="runs"):
        return self._data[endpoint]

    def getAvailFtrs(self, which="all"):
        if which == "all":
            return {key: df.columns.to_list() if isinstance(df, pd.DataFrame) else None for key, df in self._data.items()}
        elif which == "numerical":
            return {key: df.select_dtypes(include=[int, float]).columns.to_list() if isinstance(df, pd.DataFrame) else None for key, df in self._data.items()}
        elif which == "bools":
            return {key: df.select_dtypes(include=[bool]).columns.to_list() if isinstance(df, pd.DataFrame) else None for key, df in self._data.items()}
        else:
            return None
        
    def setRuns(self, runnbs: list, keep_prev=True):
        if keep_prev == False:
            self._resetRunFilters()
        for runnb in runnbs:
            self.runfilters.append(
                OMSFilter(
                    attribute_name="run_number", 
                    value=runnb,
                    operator="EQ"
                )
            )
            
    def setFilters(self, filters: dict, keep_prev=True):
        if keep_prev == False:
            self._resetFilters()
        for filter in filters:
            self.filters.append(OMSFilter(**dict))

    def fetchData(self, endpoint="runs", keep_prev=True, timeout=90):
        def query_with_timeout(endpoint, runFilter):
            try: 
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(self.dials.oms.query, endpoint=endpoint, filters=[runFilter])
                    return future.result(timeout=timeout)
            except TimeoutError:
                print("WARNING: Data fetching time out for filter {}".format(runFilter))
                return None
            except Exception as e:
                print("WARNING: Unable to fetch data for filter {}: {}".format(runFilter, e))
                return None
        
        if keep_prev == False:
            self._resetDataDict(endpoint="all")
        
        for runFilter in self.runfilters:
            print(runFilter)
            query_rslts = query_with_timeout(endpoint, runFilter)
            if query_rslts is None:
                continue
                
            query_rslts_df = makeDF(query_rslts)

            # Apply specific indexing based on endpoint
            if endpoint == "runs":
                query_rslts_df["run_number_idx"] = query_rslts_df["run_number"]
                query_rslts_df.set_index("run_number_idx", inplace=True)
                query_rslts_df.index.name = "run_number"
            elif endpoint == "lumisections":
                query_rslts_df["run_number_idx"] = query_rslts_df["run_number"]
                query_rslts_df["lumisection"] = query_rslts_df["lumisection_number"]
                query_rslts_df.set_index(["run_number_idx", "lumisection"], inplace=True)
                query_rslts_df.index.names = ["run_number", "lumisection"]

            # Combine with existing if needed
            if self._data[endpoint] is None:
                self._data[endpoint] = query_rslts_df
            else:
                self._data[endpoint] = pd.concat([self._data[endpoint], query_rslts_df])        

def makeDF(json):
    datadict = json["data"][0]["attributes"]
    keys = datadict.keys()

    datasetlist = []

    for i in range(len(json["data"])):
        values = json["data"][i]["attributes"].values()
        datasetlist.append(values)
    return pd.DataFrame(datasetlist, columns=keys)