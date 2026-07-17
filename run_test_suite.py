"""Run test.py repeatedly for multiple test sets and sampling modes.

For each data_path in DATA_PATHS and each mode in MODES, this calls test.py
once, passing the test-specific parameters (data_path, sample, num_ins) as
command-line arguments. config.json is NEVER modified, so multiple seeds can
run in parallel safely (no race condition on the shared config file).

Results for one combination are written directly into:
    ./save/<experiment.name>/test/seed<seed>/<data_path>_<greedy|sample>_<indist|ood>/

Already-finished combinations (target folder already contains an .xlsx) are
skipped, so the script can be safely re-run/resumed (e.g. after a crash on a
cluster job).
"""
import json
import os
import subprocess
import sys
import argparse

CONFIG_PATH = "./config.json"


def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model", type=str, required=True,
                        choices=["indist", "ood"],
                        help="Which best checkpoint to test: 'indist' (save_best_indist_*.pt) "
                             "or 'ood' (save_best_ood_*.pt)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Enable pooling diagnostics")
    args = parser.parse_args()

    config = load_config()
    exp_name = config["experiment"]["name"]
    save_dir = "./save/{0}/test/seed{1}".format(exp_name, args.seed)
    os.makedirs(save_dir, exist_ok=True)

    # Test sets to run (folder names under ./data_test/)
    DATA_PATHS = [
        # Synthetic instances (same structure as training)
        "1005", "1510", "2005", "2010",
        "3010", "4010", "5010", "10010", "20010",
        # More machines than training
        "2020", "2030",
        # Hurink instances (structurally different)
        "edata", "rdata", "vdata",
    ]
    # (sample, suffix): greedy is DRL-G, sample is DRL-S
    MODES = [(False, "greedy"), (True, "sample")]
    # Number of instances to use per data_path (defaults to 100 if not listed)
    NUM_INS = {
        "Mk": 10,
        "edata": 66,
        "rdata": 66,
        "vdata": 66,
    }
    DEFAULT_NUM_INS = 100

    for data_path in DATA_PATHS:
        for sample, suffix in MODES:
            target_name = "{0}_{1}_{2}".format(data_path, suffix, args.model)
            target_path = os.path.join(save_dir, target_name)
            # Skip only if this combination already produced a result file
            if os.path.isdir(target_path) and any(
                    f.endswith(".xlsx") for f in os.listdir(target_path)):
                print("Skipping {0} (already exists)".format(target_name))
                continue

            num_ins = NUM_INS.get(data_path, DEFAULT_NUM_INS)
            print("\n=== Running test: data_path={0}, sample={1}, model={2} -> {3} ===".format(
                data_path, sample, args.model, target_name))
            cmd = [sys.executable, "test.py",
                   "--seed", str(args.seed),
                   "--data_path", data_path,
                   "--sample", str(sample),
                   "--num_ins", str(num_ins),
                   "--output_dir", target_path,
                   "--model_select", args.model]
            if args.diagnose:
                cmd.append("--diagnose")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("test.py failed for {0} (exit code {1}), aborting.".format(
                    target_name, result.returncode))
                return
            print("Done {0}".format(target_name))


if __name__ == '__main__':
    main()