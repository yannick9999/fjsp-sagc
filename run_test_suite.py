"""Run test.py repeatedly for multiple test sets and sampling modes.

For each data_path in DATA_PATHS and each mode in MODES, this calls test.py
once, passing the test-specific parameters (data_path, sample, num_ins) as
command-line arguments. config.json is NEVER modified, so multiple seeds can
run in parallel safely (no race condition on the shared config file).

Results for one combination are written directly into:
    ./save/<experiment.name>/test/seed<seed>/<data_path>_<greedy|sample>/

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
    args = parser.parse_args()

    config = load_config()
    exp_name = config["experiment"]["name"]
    save_dir = "./save/{0}/test/seed{1}".format(exp_name, args.seed)
    os.makedirs(save_dir, exist_ok=True)

    # Test sets to run (folder names under ./data_test/)
    DATA_PATHS = ["20010", "Mk"]
    # (sample, suffix): greedy is DRL-G, sample is DRL-S
    MODES = [(False, "greedy")]
    # MODES = [(False, "greedy"), (True, "sample")]  # uncomment to also run sampling
    # Number of instances to use per data_path (defaults to 100 if not listed)
    NUM_INS = {
        "Mk": 10,
    }
    DEFAULT_NUM_INS = 100

    for data_path in DATA_PATHS:
        for sample, suffix in MODES:
            target_name = "{0}_{1}".format(data_path, suffix)
            target_path = os.path.join(save_dir, target_name)
            # Skip only if this combination already produced a result file
            if os.path.isdir(target_path) and any(
                    f.endswith(".xlsx") for f in os.listdir(target_path)):
                print("Skipping {0} (already exists)".format(target_name))
                continue

            num_ins = NUM_INS.get(data_path, DEFAULT_NUM_INS)
            print("\n=== Running test: data_path={0}, sample={1} -> {2} ===".format(
                data_path, sample, target_name))
            result = subprocess.run([sys.executable, "test.py",
                     "--seed", str(args.seed),
                     "--data_path", data_path,
                     "--sample", str(sample),
                     "--num_ins", str(num_ins),
                     "--output_dir", target_path])
            if result.returncode != 0:
                print("test.py failed for {0} (exit code {1}), aborting.".format(
                    target_name, result.returncode))
                return
            print("Done {0}".format(target_name))


if __name__ == '__main__':
    main()