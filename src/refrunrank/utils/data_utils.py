import os
import json
import pandas as pd
import requests
import json

def loadJSONasDF(JSONFilePath):
    """ """
    if not os.path.exists(JSONFilePath):
        raise FileNotFoundError(
            "ERROR in json_utils.py / loadjson: requested json file {} does not seem to exist...".format(
                JSONFilePath
            )
        )
    with open(JSONFilePath, "r") as f:
        JSONdict = json.load(f)
    try:
        jsondf = pd.DataFrame(JSONdict).convert_dtypes()
    except:
        jsondf = pd.DataFrame(JSONdict.items()).convert_dtypes()
    return jsondf

def makeDF(json):
    datadict = json["data"][0]["attributes"]
    keys = datadict.keys()

    datasetlist = []

    for i in range(len(json["data"])):
        values = json["data"][i]["attributes"].values()
        datasetlist.append(values)
    return pd.DataFrame(datasetlist, columns=keys) 

def loadFromWeb(url, output_file):
    try:
        # Make request and check if succesful
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the response content as JSON
            data = response.json()
            
            # Store the data as JSON
            with open(output_file, 'w') as file:
                json.dump(data, file, indent=4)
            
            print(f"Data successfully fetched and stored in {output_file}")
        else:
            print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")