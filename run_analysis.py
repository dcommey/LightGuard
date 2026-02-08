from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

# Ensure non-interactive plotting backend for scripted runs.
os.environ.setdefault("MPLBACKEND", "Agg")

from lightguard.analyzers import EnhancedDurabilityAnalyzer
from lightguard.datasets import DATASETS, get_dataset_spec
from lightguard.io import download_dataset, prepare_iscx_vpn_nonvpn_2016


def _run_diagnostics(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "Diagnostics subcommands:\n"
            "  run_analysis.py diag probe-robustness [args...]\n"
            "  run_analysis.py diag accuracy [args...]\n\n"
            "Examples:\n"
            "  .\\venv\\Scripts\\python.exe run_analysis.py diag probe-robustness --dataset darknet --probe-sizes 100,1000,5000 --reps 50 --seed 1 --output-dir results\\diagnostics\n"
            "  .\\venv\\Scripts\\python.exe run_analysis.py diag accuracy --dataset darknet --seed 1 --output-dir results\\diagnostics\n\n"
            "Run with '--help' after a subcommand to see its options."
        )
        return 0

    subcmd = argv[0]
    sub_argv = argv[1:]

    if subcmd in {"probe-robustness", "probe_size_robustness"}:
        from scripts.probe_size_robustness import main as probe_main

        return int(probe_main(sub_argv))

    if subcmd in {"accuracy", "accuracy-diagnostics", "accuracy_diagnostics"}:
        from scripts.accuracy_diagnostics import main as acc_main

        return int(acc_main(sub_argv))

    print(f"Error: unknown diagnostics subcommand '{subcmd}'")
    print("Try: run_analysis.py diag --help")
    return 2


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run LightGuard durability analysis")
    parser.add_argument(
        "--dataset",
        default="darknet",
        choices=sorted(DATASETS.keys()),
        help="Dataset preset to use.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to the dataset CSV (overrides dataset preset default).",
    )
    parser.add_argument(
        "--data-url",
        default=None,
        help=(
            "Optional direct URL to download the dataset from. "
            "If provided, this overrides the preset URL (useful for other CIC datasets)."
        ),
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if the dataset file already exists.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where datasets are stored/downloaded.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory where outputs (plots/results) are written.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated list of baseline models to run in Phase I (e.g., 'RF,XGB,BAG-DT,DNN'). "
            "If omitted, runs the full configured set."
        ),
    )
    parser.add_argument(
        "--skip-dl",
        action="store_true",
        help="Skip deep learning baselines (LSTM, Transformer) to speed up runs.",
    )
    parser.add_argument(
        "--dl-epochs",
        type=int,
        default=None,
        help="Override epochs for LSTM/Transformer baselines (if enabled).",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated list of integer seeds to run for confidence intervals "
            "(e.g., '1,2,3,4,5'). If provided, runs once per seed into subfolders "
            "output_dir/seed_<seed> and then writes aggregated 95%% CI plots to output_dir/aggregate/."
        ),
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Regenerate aggregated 95%% CI reports from existing seed_<seed> subfolders "
            "under --output-dir without rerunning experiments. "
            "Optionally combine with --seeds to select a subset."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Only download/prepare the dataset into --data-dir and then exit. "
            "Useful to verify downloads and preprocessing inputs."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Load + preprocess the dataset and then exit (no Phase I/II experiments). "
            "Implies --prepare-only is false, but will still download if missing."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0] == "diag":
        return _run_diagnostics(argv[1:])

    args = _build_arg_parser().parse_args(argv)

    if args.prepare_only and args.validate_only:
        print("Error: --prepare-only and --validate-only are mutually exclusive")
        return 2

    if args.seeds and (args.prepare_only or args.validate_only):
        print("Error: --seeds cannot be combined with --prepare-only or --validate-only")
        return 2

    if args.aggregate_only and (args.prepare_only or args.validate_only):
        print("Error: --aggregate-only cannot be combined with --prepare-only or --validate-only")
        return 2

    if args.aggregate_only and args.seeds:
        # Allowed: select subset of existing seed folders
        pass

    if args.aggregate_only and not args.seeds:
        # Allowed: auto-detect seed folders
        pass

    current_dir = Path.cwd()
    data_dir = (current_dir / args.data_dir).resolve()
    output_dir = (current_dir / args.output_dir).resolve()

    dataset_spec = get_dataset_spec(args.dataset)
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else (data_dir / dataset_spec.default_filename)

    print("LightGuard Analysis Framework")
    print("=============================")
    print(f"Dataset: {dataset_spec.display_name} ({dataset_spec.key})")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Data path: {data_path}")

    data_dir.mkdir(exist_ok=True, parents=True)
    output_dir.mkdir(exist_ok=True, parents=True)

    needs_download = args.force_download or (not data_path.exists())
    if needs_download:
        print(f"Dataset not found at {data_path}")

        download_url = args.data_url or dataset_spec.url

        if download_url:
            # Special handling: directory URL (Apache index) for ISCX-VPN ARFF zips
            if download_url.endswith("/") and args.dataset == "iscx_vpn_nonvpn_2016" and args.data_path is None:
                work_dir = data_dir / dataset_spec.key
                print(f"Downloading and preparing dataset into: {work_dir}")
                csv_dir = prepare_iscx_vpn_nonvpn_2016(download_url, work_dir)
                if csv_dir is None:
                    print("Failed to prepare ISCX-VPN-NonVPN-2016.")
                    return 1
                data_path = csv_dir
            else:
                print(f"Attempting to download from {download_url}...")
                success = download_dataset(download_url, data_path)
                if not success:
                    print("Failed to download dataset automatically.")
                    print("Please manually download the dataset from:")
                    print(download_url)
                    print("And save it to:")
                    print(data_path)
                    return 1
        else:
            print("This dataset preset does not have an automatic download URL.")
            if dataset_spec.notes:
                print(dataset_spec.notes)
            print("Provide a CSV path via --data-path, or a URL via --data-url.")
            return 1

    if args.prepare_only:
        print("\nPreparation complete.")
        print(f"Prepared data path: {data_path}")
        return 0

    print("\nInitializing Analyzer...")
    if args.aggregate_only:
        from lightguard.ci_reporting import generate_ci_reports

        seed_dirs: list[Path] = []
        if args.seeds:
            seed_list: list[int] = []
            for part in str(args.seeds).split(","):
                part = part.strip()
                if not part:
                    continue
                try:
                    seed_list.append(int(part))
                except ValueError:
                    print(f"Error: invalid seed '{part}'")
                    return 2

            if not seed_list:
                print("Error: --seeds provided but no valid seeds parsed")
                return 2

            for seed in seed_list:
                seed_output_dir = (output_dir / f"seed_{seed}").resolve()
                if not seed_output_dir.exists():
                    print(f"Error: seed directory not found: {seed_output_dir}")
                    return 2
                seed_dirs.append(seed_output_dir)
        else:
            seed_dirs = sorted([p for p in output_dir.glob("seed_*") if p.is_dir()])

        if not seed_dirs:
            print(f"Error: no seed_* directories found under: {output_dir}")
            return 2

        agg_dir = generate_ci_reports(output_dir, seed_dirs)
        print("\n" + "=" * 60)
        print("CI Aggregation Complete")
        print("=" * 60)
        print(f"Aggregated outputs saved to: {agg_dir}")
        return 0

    # Multi-seed CI mode
    if args.seeds:
        from lightguard.ci_reporting import generate_ci_reports

        seed_list: list[int] = []
        for part in str(args.seeds).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                seed_list.append(int(part))
            except ValueError:
                print(f"Error: invalid seed '{part}'")
                return 2

        if not seed_list:
            print("Error: --seeds provided but no valid seeds parsed")
            return 2

        seed_dirs: list[Path] = []
        for seed in seed_list:
            seed_output_dir = (output_dir / f"seed_{seed}").resolve()
            seed_output_dir.mkdir(exist_ok=True, parents=True)

            print(f"\n=== Running seed {seed} → {seed_output_dir} ===")
            analyzer = EnhancedDurabilityAnalyzer(data_path, seed_output_dir)
            analyzer.set_seed(seed)

            if args.models:
                requested = {m.strip().upper() for m in args.models.split(",") if m.strip()}
                analyzer.enabled_model_names = requested
                print(f"Using models: {sorted(requested)}")

            if args.skip_dl:
                analyzer.include_dl_models = False

            if args.dl_epochs is not None:
                if args.dl_epochs <= 0:
                    print("Error: --dl-epochs must be > 0")
                    return 2
                analyzer.dl_epochs = int(args.dl_epochs)

            print("\n=== Phase I: Diagnosing the Durability Deception ===")
            if analyzer.run_full_analysis():
                print("\n=== Phase II: LightGuard Solution ===")
                if analyzer.initialize_lightguard(base_model_name="XGB", buffer_size_percent=5.0):
                    analyzer.evaluate_lightguard_longitudinal()
                    analyzer.visualize_lightguard_performance()
                    analyzer.visualize_tradeoff_analysis()
                    analyzer.run_ablation_study()

                    print("\n=== Running Drift Threshold Ablation Study ===")
                    analyzer.run_drift_threshold_ablation()

                    print("\n=== Running Continual Learning Baselines Comparison ===")
                    analyzer.run_continual_learning_comparison()
            else:
                print(f"Seed {seed}: Phase I analysis failed.")

            seed_dirs.append(seed_output_dir)

        agg_dir = generate_ci_reports(output_dir, seed_dirs)
        print("\n" + "=" * 60)
        print("CI Aggregation Complete")
        print("=" * 60)
        print(f"Aggregated outputs saved to: {agg_dir}")
        return 0

    analyzer = EnhancedDurabilityAnalyzer(data_path, output_dir)

    if args.models:
        requested = {m.strip().upper() for m in args.models.split(",") if m.strip()}
        analyzer.enabled_model_names = requested
        print(f"Using models: {sorted(requested)}")

    if args.skip_dl:
        analyzer.include_dl_models = False

    if args.dl_epochs is not None:
        if args.dl_epochs <= 0:
            print("Error: --dl-epochs must be > 0")
            return 2
        analyzer.dl_epochs = int(args.dl_epochs)

    if args.validate_only:
        print("\nValidating dataset (load + preprocess only)...")
        ok = analyzer.load_and_preprocess_data()
        print("Validation OK" if ok else "Validation FAILED")
        return 0 if ok else 1

    print("\n=== Phase I: Diagnosing the Durability Deception ===")
    if analyzer.run_full_analysis():
        print("\n=== Phase II: LightGuard Solution ===")
        if analyzer.initialize_lightguard(base_model_name="XGB", buffer_size_percent=5.0):
            analyzer.evaluate_lightguard_longitudinal()
            analyzer.visualize_lightguard_performance()
            analyzer.visualize_tradeoff_analysis()
            analyzer.run_ablation_study()

            print("\n=== Running Drift Threshold Ablation Study ===")
            analyzer.run_drift_threshold_ablation()

            print("\n=== Running Continual Learning Baselines Comparison ===")
            analyzer.run_continual_learning_comparison()

            print("\n" + "=" * 60)
            print("Analysis Complete!")
            print("=" * 60)
            print(f"\nAll results saved to: {output_dir}")
    else:
        print("Phase I analysis failed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
