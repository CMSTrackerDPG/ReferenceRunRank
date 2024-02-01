## Reference Run Ranking with PCA

### Overview
This project focuses on developing tools to automate the reference run selection for data quality monitoring for the Tracker system at CMS. It uses Principal Component Analysis as its primary tool to rank a set of candidate runs by how similar in data taking conditions they are to a particular target run one wishes to find a referece run to.

### Ranking
- **PCA-Based Ranking**: Implements PCA to transform feature space, capturing the most significant patterns differentiating runs.
- **Distance Measurement**: Ranks runs by calculating the Euclidean distance in the PCA-transformed space, ensuring that runs closer to the target in this space are ranked higher.
- **Pre-Run Filtering**: Considers only runs occurring before the target run for ranking.
- **Standardization Option**: Incorporates data standardization to treat all features equally, removing bias due to differing scales.
- **Original Data Preservation**: Offers an option to retain original, unstandardized feature data alongside PCA results for comprehensive analysis.

### Contributions and Feedback
Contributions to this project are welcome. Feel free to fork the repository, submit pull requests, or provide feedback and suggestions through issues.
