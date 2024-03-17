# Reference Run Ranking with PCA

## Overview
This project focuses on developing tools to automate the reference run selection for data quality monitoring for the Tracker system at CMS. It uses Principal Component Analysis as its primary tool to rank a set of candidate runs by how similar in data taking conditions they are to a particular target run one wishes to find a referece run to.

## Ranking
- **PCA-Based Ranking**: Implements PCA to transform feature space, capturing the most significant patterns differentiating runs.
- **Distance Measurement**: Ranks runs by calculating the Euclidean distance in the PCA-transformed space, ensuring that runs closer to the target in this space are ranked higher.
- **Pre-Run Filtering**: Considers only runs occurring before the target run for ranking.
- **Standardization Option**: Incorporates data standardization to treat all features equally, removing bias due to differing scales.
- **Original Data Preservation**: Offers an option to retain original, unstandardized feature data alongside PCA results for comprehensive analysis.

## Concept
This tool will be implemented in Dials. The idea behind the webpage is the following:
- Webpage where shifter/shift leader can input 
    - The run they want to find a good reference run for
    - Basic requirements for their reference run
    - The runs that they want to consider as candidate reference runs
    - The features they want to be considered in the ranking process
    - Dataset
    - Other advanced options
        - Number of components in PCA (default = 2)
        - Min number of LSs (default = 600)
        - Extra filters on candidate runs
- Output is a table with
    - Candidate runs ordered by ranking
    - Number of LSs of each run
    - Quick link to OMS and GUI
    - Value of selected features
- Run number of candidate runs is clickable and clicking it opens a side window which contains the following for the run
    - In/Out components
    - Full OMS metadata
    - Certification notes left by shifter (fetched form Cert Helper)
    - Summary plots
    - Search results for the selected run number (elogs might not have proper API to be able to do this)
- Ranking of runs will be downloadable/saveable.
- Tool will be available in the ML Playground

## Contributions and Feedback
Contributions to this project are welcome. Feel free to fork the repository, submit pull requests, or provide feedback and suggestions through issues.
