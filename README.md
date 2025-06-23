# Reference Run Ranking

## Setup

To install Reference Run Ranking (RRR), run the following in your virtual environment.

```bash
pip3 install git+https://github.com/CMSTrackerDPG/ReferenceRunRank.git
```

## Running Reference Run Ranking

### Command Line

You can run RRR from your terminal by using the command `rrr`. The basic requirements to run the script are:

* RRR configuration JSON
    - Example: [`configs/refrank_config_example.json`](configs/refrank_config_example.json)
    - In this configuration file, you can specify the features that you want to use for the ranking, as well as the runs that will be fetched from OMS and be treated as targets, and filters on those runs.
* "Golden" JSON (Optional)
    - Can be an official, or a user defined golden JSON. It can be fetched through the [JSON portal](https://cmsrunregistry.web.cern.ch/json_portal) in Run Registry, or by using the command `fetch_golden` from DQM Explore ([documentation]()).
* Target run:
    - You will need to provide the run number of the run you want to find a reference run for. This number does not need to be included in the RRR configuration JSON.

Basic example usage:
```bash
rrr --config configs/refrank_config.json --golden ./jsons/golden_PromptReco-Collisions2024-2025.json --rslts_fname ./rankings/rankings_385312.json --target 385312
```

### Notebook

An example notebook is included for running the ranking algorithm and/or loading results JSON files to study the results closer. This notebook can be found in [`notebooks/RefRunRank.ipynb`](notebooks/RefRunRank.ipynb)
