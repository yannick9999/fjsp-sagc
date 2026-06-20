import argparse
import copy
import json
import os
import random
import time as time

import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model
from env.fjsp_env import FJSPEnv
from env.load_data import nums_detec

def format_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "{0:02d}:{1:02d}:{2:02d}".format(h, m, s)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config.json (also selects model_dir)")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Test set folder under ./data_test/ (e.g. 2010, Mk)")
    parser.add_argument("--sample", type=lambda x: str(x).lower() == "true", default=False,
                        help="DRL-S sampling mode (true/false). Default false = greedy DRL-G")
    parser.add_argument("--num_ins", type=int, required=True,
                        help="Number of instances to evaluate")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write test results into")
    args = parser.parse_args()

    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    handle = None
    if device.type == 'cuda':
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')
    else:
        torch.set_default_dtype(torch.float32)
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    # peak GPU memory
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    exp_name = load_dict["experiment"]["name"]
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"]
    # CLI overrides for test-specific parameters (kept out of config.json)
    seed = args.seed if args.seed is not None else train_paras["seed"]
    test_paras["data_path"] = args.data_path
    test_paras["sample"] = args.sample
    test_paras["num_ins"] = args.num_ins
    setup_seed(seed)
    print(f"Seed: {seed}")
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    # Model directory derived from experiment name + seed (single source of truth)
    model_dir = './save/{0}/seed{1}'.format(exp_name, seed)

    data_path = "./data_test/{0}/".format(test_paras["data_path"])
    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: x[:-4])
    test_files = test_files[:num_ins]
    mod_files = os.listdir(model_dir)[:]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras)
    rules = test_paras["rules"]
    envs = []  # Store multiple environments
    # coarsening overhead + peak GPU memory setup
    has_graph_unet = hasattr(model.policy_old, 'graph_unet')
    if has_graph_unet:
        model.policy_old.graph_unet._timing_enabled = True
    coarsening_rows = []
    ckpt_pooling_cfg = None  # filled from checkpoint, used for metadata sheet

    # Detect and add models to "rules"
    if "DRL" in rules:
        for root, ds, fs in os.walk(model_dir):
            for f in fs:
                if f.endswith('.pt'):
                    rules.append(f)
    if len(rules) != 1:
        if "DRL" in rules:
            rules.remove("DRL")

    # Output paths (write results directly into output_dir)
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = args.output_dir
    os.makedirs(save_path, exist_ok=True)

    file_name_col = [test_files[i] for i in range(num_ins)]

    # Accumulators: dict of rule -> list of values (one per instance)
    makespan_by_rule = {}
    time_by_rule = {}

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            ckpt_path = os.path.join(model_dir, mod_files[i_rules])
            if device.type == 'cuda':
                model_CKPT = torch.load(ckpt_path)
            else:
                model_CKPT = torch.load(ckpt_path, map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])

            if isinstance(model_CKPT, dict) and "state_dict" in model_CKPT:
                state_dict = model_CKPT["state_dict"]
                ckpt_model_paras = model_CKPT["model_paras"]
                ckpt_model_paras["device"] = device
                ckpt_model_paras["actor_in_dim"] = ckpt_model_paras["out_size_ma"] * 2 + ckpt_model_paras["out_size_ope"] * 2
                ckpt_model_paras["critic_in_dim"] = ckpt_model_paras["out_size_ma"] + ckpt_model_paras["out_size_ope"]
                model = PPO_model.PPO(ckpt_model_paras, train_paras)
                model.policy.to(device)
                model.policy_old.to(device)
                if hasattr(model.policy_old, 'graph_unet'):
                    model.policy_old.graph_unet._timing_enabled = True
                has_graph_unet = hasattr(model.policy_old, 'graph_unet')
                ckpt_pooling_cfg = ckpt_model_paras.get("pooling", {})
                print(f"  config from checkpoint: method={ckpt_model_paras['pooling']['method']}")
            else:
                state_dict = model_CKPT
                print("  WARNING: old checkpoint format, using config.json for architecture")

            model.policy.load_state_dict(state_dict)
            model.policy_old.load_state_dict(state_dict)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            with open(test_file) as file_object:
                line = file_object.readlines()
                ins_num_jobs, ins_num_mas, _ = nums_detec(line)
            env_test_paras["num_jobs"] = ins_num_jobs
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
            # Create environment object
            else:
                # Clear the existing environment
                if handle is not None:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if meminfo.used / meminfo.total > 0.7:
                        envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance
                if test_paras["sample"]:
                    env = FJSPEnv(case=[test_file] * test_paras["num_sample"],
                                  env_paras=env_test_paras, data_source='file')
                # DRL-G, each env contains one instance
                else:
                    env = FJSPEnv(case=[test_file], env_paras=env_test_paras, data_source='file')
                envs.append(copy.deepcopy(env))
                print("Create env[{0}]".format(i_ins))

            # Schedule an instance/environment
            if has_graph_unet:
                model.policy_old.graph_unet.reset_timing()
            # DRL-S
            if test_paras["sample"]:
                makespan, time_re = schedule(env, model, memories, flag_sample=test_paras["sample"])
                makespans.append(torch.min(makespan).item())
                times.append(time_re)
            # DRL-G
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                for j in range(test_paras["num_average"]):
                    makespan, time_re = schedule(env, model, memories)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    env.reset()
                makespans.append(torch.mean(torch.tensor(makespan_s)).item())
                times.append(torch.mean(torch.tensor(time_s)).item())
            if has_graph_unet:
                _stats = model.policy_old.graph_unet.get_timing_stats()
                _n_dec = model.policy_old.graph_unet._n_calls
            else:
                _stats = {"avg_coarse_ms": 0.0, "avg_forward_ms": 0.0, "overhead_pct": 0.0}
                _n_dec = 0
            coarsening_rows.append({
                "file_name": test_files[i_ins],
                "rule": rule,
                "avg_coarse_ms": _stats["avg_coarse_ms"],
                "avg_forward_ms": _stats["avg_forward_ms"],
                "overhead_pct": _stats["overhead_pct"],
                "n_decisions": _n_dec,
            })
            elapsed = time.time() - start
            completed = i_rules * num_ins + i_ins + 1
            total_to_do = len(rules) * num_ins
            avg_per_instance = elapsed / completed
            estimated_total = avg_per_instance * total_to_do
            remaining = estimated_total - elapsed
            print("finish env {0} | elapsed: {1} | estimated total: {2} | remaining: ~{3}".format(
                i_ins, format_time(elapsed), format_time(estimated_total), format_time(remaining)))
        print("rule_spend_time: ", time.time() - step_time_last)

        makespan_by_rule[rule] = makespans
        time_by_rule[rule] = times

        for env in envs:
            env.reset()

    total_runtime_sec = time.time() - start

    # Peak GPU memory over the full test run
    if device.type == 'cuda':
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gpu_gb = '{:.4f}'.format(peak_bytes / (1024 ** 3))
    else:
        peak_gpu_gb = 'N/A'

    # Build DataFrames for the four sheets
    df_makespan = pd.DataFrame({"file_name": file_name_col})
    for rule in rules:
        df_makespan[rule] = makespan_by_rule[rule]

    df_time = pd.DataFrame({"file_name": file_name_col})
    for rule in rules:
        df_time[rule] = time_by_rule[rule]

    df_coarsening = pd.DataFrame(coarsening_rows)

    # Pooling info for metadata comes from the checkpoint (the true architecture),
    # falling back to config.json only for old-format checkpoints.
    pooling_cfg = ckpt_pooling_cfg if ckpt_pooling_cfg is not None else model_paras.get("pooling", {})
    metadata = [
        ("experiment",      exp_name),
        ("timestamp",       str_time),
        ("seed",            str(seed)),
        ("method",          str(pooling_cfg.get("method", ""))),
        ("num_pool_layers", str(pooling_cfg.get("num_layers", ""))),
        ("pool_ratio",      str(pooling_cfg.get("ratio", ""))),
        ("num_jobs",        str(env_paras["num_jobs"])),
        ("num_mas",         str(env_paras["num_mas"])),
        ("data_path",       str(test_paras["data_path"])),
        ("num_ins",         str(test_paras["num_ins"])),
        ("sample",          str(test_paras["sample"])),
        ("num_sample",      str(test_paras["num_sample"]) if test_paras["sample"] else ""),
        ("device",          device.type),
        ("peak_gpu_gb",     peak_gpu_gb),
        ("total_runtime_sec", str(round(total_runtime_sec, 2))),
    ]
    df_meta = pd.DataFrame(metadata, columns=["key", "value"])

    # Write all sheets into a single Excel file
    out_path = '{0}/test_results_{1}.xlsx'.format(save_path, str_time)
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_makespan.to_excel(writer, sheet_name='makespan', index=False)
        df_time.to_excel(writer, sheet_name='solve_time', index=False)
        df_coarsening.to_excel(writer, sheet_name='coarsening_overhead', index=False)
        df_meta.to_excel(writer, sheet_name='run_metadata', index=False)

    print("total_spend_time: ", total_runtime_sec)

def schedule(env, model, memories, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    i = 0
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    # print("spend_time: ", spend_time)

    # Verify the solution
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    return copy.deepcopy(env.makespan_batch), spend_time


if __name__ == '__main__':
    main()