# Reference Run Ranking

## Overview

This project focuses on developing tools to automate the reference run selection for data quality monitoring at CMS. It uses Principal Component Analysis as its primary tool to rank a set of candidate runs by how similar in data taking conditions they are to a particular target run one wishes to find a referece run for.

## Concept

The base algorithm uses the data taking conditions (luminosity, fill number, etc.) as features to find the principal components for a set of given runs (candidates + target) and projects these runs unto this PCA subspace. With the reduced dimensionality, candidate runs are ranked according to the Euclidian distance from the target run. One can also use hierarchical clustering before the ranking in order to automatically have the least correlated features selected used.

## Contributions and Feedback

This project was originally worked on by Guillermo Fidalgo.