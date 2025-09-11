# Reference Run Ranking

## Setup
To use Reference Run Ranking (RRR), you must install it and its dependencies in a Python virtual environment. A script is included in this repository that will set up a clean environment and make the necessary installations. The following commands will download this script and run it:

```bash
wget https://raw.githubusercontent.com/CMSTrackerDPG/ReferenceRunRank/main/scripts/setup.sh && chmod +x setup.sh && ./setup.sh
```

If you already have a virtual environment, you can set up RRR by running the following command to add RRR to it:

```bash
pip3 install git+https://github.com/CMSTrackerDPG/ReferenceRunRank.git
```

## Running Reference Run Ranking

### Command Line

You can run RRR from your terminal by using the command `rrr`. The basic requirements to run this are:

* RRR configuration JSON
    - This is where you specify the features that you want to use for the ranking, as well as the runs that will be fetched from OMS and be treated as targets, and filters on those runs.
    - Example: [`configs/refrank_config_example.json`](configs/refrank_config_example.json)
* "Golden" JSON or configuration file to generate one (Optional)
    - Can be an official, or a user defined golden JSON. There are multiple ways to generate your own Golden JSON:
        - The argument `--golden_config` accepts a path to a configuration JSON which contains the logic to auto-generate a golden JSON when running RRR. A simple example configuration is included in [`configs/runreg_logic.json`](configs/runreg_logic.json). Note that to use this option you will need a CERN SSO client ID  and secret (instructions on how to obtain these are linked in [Relevant Documentation](#relevant-documentation)). You will need to add these to the included template [.env](.env) file.
        - [DQM Explore](https://github.com/CMSTrackerDPG/DQMExplore) includes the command `fetch_golden` with which you can generate your own golden JSON from the command line.
        - You can use the [JSON portal](https://cmsrunregistry.web.cern.ch/json_portal) in Run Registry to create your own golden JSON.
* Target run:
    - You will need to provide the run number of the run you want to find a reference run for. It is not necessary to include this number in the RRR configuration JSON.

#### Basic example usage

```bash
rrr --config configs/refrank_config.json --golden ./jsons/golden_PromptReco-Collisions2024-2025.json --rslts_fname ./rankings/rankings_385312.json --target 385312
```

To auto-generate the golden JSON, change `--golden` to `--golden_config` and provide the path to the appropriate configuration JSON. You can also add `--dump_golden` to save the generated golden JSON to a file.

```bash
rrr --config configs/refrank_config.json --rslts_fname ./rankings/rankings_385312.json --target 385312 --golden_config ./configs/runreg_logic.json --dump_golden
```


### Notebook

An example notebook is included for running the ranking algorithm and/or loading results JSON files to study the results closer. This notebook can be found in [`notebooks/RefRunRank.ipynb`](notebooks/RefRunRank.ipynb).

## Relevant Documentation & Resources

* [CERN SSO registration instructions](https://github.com/CMSTrackerDPG/cernrequests#for-cern-apis-using-the-new-sso)
* [DIALS Python API repository](https://gitlab.cern.ch/cms-dqmdc/libraries/dials-py/-/tree/develop)
* [Run Registry Python API repository](https://gitlab.cern.ch/cms-dqmdc/libraries/runregistry_api_client)
