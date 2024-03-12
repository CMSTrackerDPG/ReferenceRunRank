from numpy import trapz
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rank_utils import refrank

def test_rank(targets_ftrs, candidates_ftrs, RRs_used, comp_number=60, n_components=2, print_stats=True, plot=True, dpi=150, return_RRranks=True):
    """
    Iterates over a set of given target runs and performs PCA ranking. Then performs statistics on the results by evaluating 
    the rank that the actual reference run of each target run got.

    Args:
        targets: Dataframe containing the target runs in the ranking test along with their features.
        candidates: Dataframe containing the set of candidate runs which will be used in the testing along with their features
        comp_number: Comparison number; the number of candaidate runs in the past with respect to the target that should be considered candidates
        n_components: PCA dimensionality
        print_stats: Print statistics of the testing
        plot: Plot CDF plot
        dpi: DPI of CDF plot
        return_RRranks: Return the ranking results

    Returns:
        Histogram of the ranks the actual reference runs used got and (optional) the results of the ranking.
    """
    
    RRresults = {"runs":[], "RRs": [], "ranks": []}
    
    for i, (index, row) in enumerate(targets_ftrs.iterrows()):
        target_df = pd.DataFrame(row).T
        candidates = candidates_ftrs.loc[:index-1]
        if len(candidates) > comp_number:
            rank_df = refrank(target_df, candidates.loc[:target_df.index[0] - 1].iloc[-comp_number:], n_components=n_components)
            RR_used = RRs_used[index]
        else:
            # print("Not enough candidates. Skipping.")
            continue

        try: 
            actualRR_rank = rank_df[rank_df.index.get_level_values("run") == RR_used].index[0][0]
            RRresults["runs"].append(index)
            RRresults["RRs"].append(RR_used)
            RRresults["ranks"].append(actualRR_rank)
        except:
            # print("Failed to fetch actual RR used")
            continue
    
    RRresults = pd.DataFrame(RRresults)
    RRresults.sort_values("ranks", inplace=True)

    # print(RRresults)
    
    if print_stats:
        # Lower mean rank indicates better performance
        print("Mean rank of actual RR: {}".format(np.mean(RRresults["ranks"])))
        # Less sensitive to outliers than mean rank, better idea of central tendency
        print("Median rank of actual RR: {}".format(np.median(RRresults["ranks"])))
        # Measures how often actual RR appears withing top-k ranks
        print("Top-k accuracy (k=10): {}".format(sum(rank < 10 for rank in RRresults["ranks"]) / len(RRresults["ranks"])))
        # Stat measure for evaluating processes that produce a list of possible responses to a sample of queries, ordered by probability of correctness. 
        # Its the average of the reciprocal ranks of results for a sample of queries
        print("Mean reciprocal rank: {}".format(np.mean([1.0 / (rank + 1) for rank in RRresults["ranks"]])))
        
        # Making CDF plot
        
        RRranks_sorted = RRresults["ranks"].to_numpy()
        
        x_normalized = RRranks_sorted / RRranks_sorted.max()
        cdf = np.arange(1, len(RRranks_sorted) + 1) / len(RRranks_sorted)
        
        auc = trapz(cdf, x_normalized)
        
        plt.plot(RRranks_sorted, cdf)
        plt.xlabel("Rank of Actual RR")
        plt.ylabel("CDF")
        plt.title("CDF of Ranks of Actual RRs")
        plt.text(0.95, 0.05, f"AUC (normalized): {auc:.2f}", ha='right', va='bottom', transform=plt.gca().transAxes, fontsize=15, bbox=dict(facecolor='white', alpha=0.5))
        plt.grid(True)
        plt.show()
    
    fig, ax = plt.subplots(dpi=dpi)
    ax.hist(RRresults["ranks"], bins=25)
    ax.set_title("RRR results for n={}, comparison_num={}".format(n_components, comp_number))
    ax.set_xlabel("Rank of actual reference run")

    ax.axvline(x=10, color="r", linestyle="--", linewidth=2)

    plt.show()
    
    if return_RRranks:
        return RRresults.reset_index().drop("index", axis=1)
    