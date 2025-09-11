import argparse
import refrunrank as rrr
import json
import dqmexplore as dqme
import runregistry

EPs = set(["runs", "lumisections"])


def main():
    parser = argparse.ArgumentParser(description="Reference run ranking")

    parser.add_argument("--target", type=int, default=None, help="Target run number")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rrr_config.json",
        help="Configuration JSON path",
    )
    # Run Registry
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--golden", type=str, default=None, help="Golden JSON path")
    group.add_argument(
        "--golden_config", type=str, default=None, help="Golden JSON config path"
    )
    parser.add_argument(
        "--golden_config_dataset",
        type=str,
        default="/Express/Collisions2024/DQM",
        help="Dataset to be used when fetching golden JSON using config",
    )
    # Argument to dump generated golden JSON. SHould only be used with --golden_config. Argument is either true or false, with the default being false
    parser.add_argument(
        "--dump_golden", action="store_true", help="Dump generated golden JSON"
    )

    # Ranking options
    parser.add_argument(
        "--n_components", type=int, default=4, help="Number of PCA components to use"
    )
    parser.add_argument(
        "--rslts_fname", type=str, default="rrr_rslts.json", help="RRR results json"
    )
    parser.add_argument("--wghts", action="store_true", help="Output weights")
    parser.add_argument(
        "--keep_ftrs", action="store_false", help="Keep unscaled features in the output"
    )
    args = parser.parse_args()

    dials = dqme.utils.setupdials.setup_dials_object_deviceauth()
    omsdata = dqme.omsdata.OMSData(dials)
    omsdata.setFilters(args.config)

    # Open rrr_config.json
    with open(args.config) as f:
        config = json.load(f)

    # Getting the endpoints to be used for ranking
    endpoints = set()
    for ftr_type in ["ftrs", "pca_ftrs"]:
        if len(config.get(ftr_type, {})) > 0:
            ftrs = config[ftr_type]
            # Do intersection between EPs and ftrs keys. Then, add the result to endpoints set
            endpoints = endpoints.union(set(ftrs.keys()).intersection(EPs))

    if len(endpoints) == 0:
        raise ValueError("No valid endpoints found in the configuration file.")
    print("Endpoints to be used for ranking: {}".format(endpoints))

    # Making sure runs endpoint is first
    if "runs" in endpoints:
        endpoints.remove("runs")
        endpoints = ["runs"] + list(endpoints)

    for endpoint in endpoints:
        omsdata.fetchData(
            endpoint=endpoint,
            include=[args.target] if endpoint == "runs" else None,
            match_runs=(endpoint == "lumisections"),
        )

    # Check if data is fetched
    if omsdata._data is None:
        raise ValueError(
            "No data fetched from OMS. Please check your configurations and endpoints."
        )

    # Apply golden JSON if provided
    if args.golden is not None:
        omsdata.applyGoldenJSON(args.golden, keep=[args.target])
    elif args.golden_config is not None:
        with open(args.golden_config) as f:
            golden_config = json.load(f)
        print(
            f"Generating golden JSON using dataset:\n   Dataset:{args.golden_config_dataset}\n   Logic: {args.golden_config}"
        )

        rr_data = runregistry.create_json(
            json_logic=golden_config,
            dataset_name_filter=golden_config.get(
                "dataset", args.golden_config_dataset
            ),
        )["generated_json"]
        if args.dump_golden:
            with open("generated_golden.json", "w") as gf:
                json.dump(rr_data, gf, indent=4)
            print("Generated golden JSON dumped to generated_golden.json")
        omsdata.applyGoldenJSON(rr_data, keep=[args.target])
    else:
        print("WARNING: No golden JSON provided. Proceeding without it.")

    # Check if any runs are left after applying golden JSON
    if len(omsdata._data["runs"]) <= 1:
        raise ValueError(
            "No runs left after applying golden JSON. Please check your golden JSON or the configuration you used to generate it."
        )

    ranker = rrr.ranking.RunRanker(omsdata, ftrs=config)
    ranker.constructFeatures()

    rankings, wghts, _, _ = ranker.refrank(
        args.target,
        n_components=args.n_components,
        keep_ftrs=args.keep_ftrs,
    )

    # Output results to file
    print("Results output to {}".format(args.rslts_fname))
    if args.rslts_fname.endswith(".json"):
        rankings.to_json(args.rslts_fname, orient="index", indent=4, index=True)
    elif args.rslts_fname.endswith(".csv"):
        rankings.to_csv(args.rslts_fname, index=True)
    else:
        print("Invalid file format. Please use .json or .csv")
        return

    if args.wghts:
        import pandas as pd

        wghts_fname = args.rslts_fname.replace(".json", "_weights.json").replace(
            ".csv", "_weights.csv"
        )
        wghts_df = pd.DataFrame(wghts)
        # Add column indicating which PC the weight belongs to, starting from 1, putting it at the first column
        wghts_df["PC"] = [i + 1 for i in range(args.n_components)]
        cols = list(wghts_df.columns)
        wghts_df = wghts_df[[cols[-1]] + cols[:-1]]
        wghts_df.to_json(wghts_fname, orient="records", indent=4)


if __name__ == "__main__":
    main()
