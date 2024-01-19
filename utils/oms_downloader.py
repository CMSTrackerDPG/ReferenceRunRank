# local modules
import get_oms_data
from get_oms_data import get_oms_api, get_oms_data, get_oms_response_attribute

import pandas as pd
import numpy as np
import json

class oms_downloader:
    def __init__(self, run_start, run_end):
        # initialize oms api and store it to use it in get_oms_data
        self.omsapi = get_oms_api
    
    def run_download(self, limit_entries, attributes, ):
        # download run data and save it in file; if limit_entries
        
    

