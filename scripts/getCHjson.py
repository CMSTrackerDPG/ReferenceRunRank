from refrunrank.utils.data_utils import loadFromWeb
import argparse

parser = argparse.ArgumentParser(description="A script to get CertHelper data downloaded into a JSON file")
parser.add_argument("--fname", type=str, default="./jsons/ch_refruns.json", help="Name and path of output file.")
args = parser.parse_args()

print("Downloading CH run data and storing it in ch_refruns.json")

url = "https://certhelper.web.cern.ch/certify/allRunsRefRuns/"

loadFromWeb(url, args.fname)