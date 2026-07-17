import argparse
import copy
import json
import os
import random
import shutil
import time
from collections import deque

import pandas as pd
import torch
import numpy as np

import PPO_model
from env.case_generator import CaseGenerator
from env.fjsp_env import FJSPEnv
from validate import validate, get_validate_envs


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def save_training_excel(excel_path, valid_iterations, env_steps_at_valid,
                        valid_indist_norm, valid_ood_norm, metadata_rows):
    """Write training results to Excel. Uses atomic write to prevent corruption."""
    import tempfile

    df_curve = pd.DataFrame({
        'iteration': valid_iterations,
        'env_steps': env_steps_at_valid,
        'indist_norm': valid_indist_norm,
        'ood_norm': valid_ood_norm,
    })

    df_meta = pd.DataFrame(metadata_rows, columns=["key", "value"])

    # Write to a temp file first, then atomically replace
    dir_name = os.path.dirname(excel_path)
    with tempfile.NamedTemporaryFile(dir=dir_name, suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            df_curve.to_excel(writer, sheet_name='validation_curve', index=False)
            df_meta.to_excel(writer, sheet_name='run_metadata', index=False)
        os.replace(tmp_path, excel_path)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None,
                        help="Override seed from config.json")
    args = parser.parse_args()

    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_dtype(torch.float32)
        torch.set_default_device('cuda')
    else:
        torch.set_default_dtype(torch.float32)
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    # Load config and init objects
    with open("./config.json", 'r') as load_f:
        load_dict = json.load(load_f)
    exp_name = load_dict["experiment"]["name"]
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    train_sizes = train_paras["train_sizes"]
    size_configs = [(s["num_jobs"], s["num_mas"]) for s in train_sizes]
    size_weights = [s["weight"] for s in train_sizes]
    if args.seed is not None:
        train_paras["seed"] = args.seed
    seed = train_paras["seed"]
    setup_seed(seed)
    print(f"Seed: {seed}")
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    memories = PPO_model.Memory()
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    valid_envs = get_validate_envs(env_valid_paras)  # Create environments for validation
    maxlen = 1  # Save the best model per category
    best_models_indist = deque()
    best_models_ood = deque()
    makespan_best_indist = float('inf')
    makespan_best_ood = float('inf')

    # Output paths
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/{0}/seed{1}'.format(exp_name, seed)
    os.makedirs(save_path, exist_ok=True)
    # Snapshot the exact config used for this run (audit trail against later edits)
    shutil.copy("./config.json", os.path.join(save_path, "config_used.json"))
    excel_path = '{0}/train_results_{1}.xlsx'.format(save_path, str_time)

    def build_metadata(peak_gpu_gb='N/A', total_runtime_sec='in_progress'):
        pooling_cfg = model_paras.get("pooling", {})
        return [
            ("experiment",      exp_name),
            ("timestamp",       str_time),
            ("seed",            str(seed)),
            ("method",          str(pooling_cfg.get("method", ""))),
            ("num_pool_layers", str(pooling_cfg.get("num_layers", ""))),
            ("pool_ratio",      str(pooling_cfg.get("ratio", ""))),
            ("train_sizes",     str([(s["num_jobs"], s["num_mas"], s["weight"]) for s in train_sizes])),
            ("batch_size",      str(env_paras["batch_size"])),
            ("max_iterations",  str(train_paras["max_iterations"])),
            ("device",          device.type),
            ("peak_gpu_gb",     str(peak_gpu_gb)),
            ("total_runtime_sec", str(total_runtime_sec)),
        ]

    # Accumulators for validation results
    valid_iterations = []
    valid_indist_norm = []
    valid_ood_norm = []
    total_env_steps = 0
    env_steps_at_valid = []

    # Start training iteration
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"]+1):
        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # Sample an instance size for this block
            (num_jobs, num_mas) = random.choices(size_configs, weights=size_weights, k=1)[0]
            env_paras["num_jobs"] = num_jobs
            env_paras["num_mas"] = num_mas
            opes_per_job_min = int(num_mas * 0.8)
            opes_per_job_max = int(num_mas * 1.2)
            # \mathcal{B} instances use consistent operations to speed up training
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            env = FJSPEnv(case=case, env_paras=env_paras)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        steps_this_episode = 0
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            steps_this_episode += 1
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        total_env_steps += env_paras["batch_size"] * steps_this_episode
        print("spend_time: ", time.time()-last_time)

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % train_paras["update_timestep"] == 0:
            loss, reward = model.update(memories, env_paras, train_paras)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % train_paras["save_timestep"] == 0:
            elapsed = time.time() - start_time
            remaining = (elapsed / i) * (train_paras["max_iterations"] - i)
            total = (elapsed / i) * train_paras["max_iterations"]
            print(f'\n[Iter {i}/{train_paras["max_iterations"]} | '
                  f'Elapsed: {time.strftime("%H:%M:%S", time.gmtime(elapsed))} | '
                  f'Remaining: ~{time.strftime("%H:%M:%S", time.gmtime(remaining))} | '
                  f'Total: ~{time.strftime("%H:%M:%S", time.gmtime(total))}]')
            print('Start validating')
            # Record the average results and the results on each instance
            indist_score, ood_score = validate(valid_envs, model.policy_old)
            valid_iterations.append(i)
            valid_indist_norm.append(indist_score)
            valid_ood_norm.append(ood_score)
            env_steps_at_valid.append(total_env_steps)

            # Save the best model per validation category (indist / ood)
            if indist_score < makespan_best_indist:
                makespan_best_indist = indist_score
                if len(best_models_indist) == maxlen:
                    delete_file = best_models_indist.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_indist_{1}.pt'.format(save_path, i)
                best_models_indist.append(save_file)
                checkpoint = {
                    "state_dict": model.policy.state_dict(),
                    "model_paras": {k: v for k, v in model_paras.items() if k != "device"},
                    "env_paras": {k: v for k, v in env_paras.items() if k != "device"},
                }
                torch.save(checkpoint, save_file)

            if ood_score < makespan_best_ood:
                makespan_best_ood = ood_score
                if len(best_models_ood) == maxlen:
                    delete_file = best_models_ood.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_ood_{1}.pt'.format(save_path, i)
                best_models_ood.append(save_file)
                checkpoint = {
                    "state_dict": model.policy.state_dict(),
                    "model_paras": {k: v for k, v in model_paras.items() if k != "device"},
                    "env_paras": {k: v for k, v in env_paras.items() if k != "device"},
                }
                torch.save(checkpoint, save_file)

            # Save Excel after every validation (crash-safe)
            elapsed_so_far = time.time() - start_time
            if device.type == 'cuda':
                peak_bytes = torch.cuda.max_memory_allocated(device)
                current_peak_gpu = '{:.4f}'.format(peak_bytes / (1024 ** 3))
            else:
                current_peak_gpu = 'N/A'
            metadata_rows = build_metadata(peak_gpu_gb=current_peak_gpu,
                                            total_runtime_sec=round(elapsed_so_far, 2))
            save_training_excel(excel_path, valid_iterations, env_steps_at_valid,
                                valid_indist_norm, valid_ood_norm, metadata_rows)

    # Final save after training completes
    total_runtime_sec = time.time() - start_time
    if device.type == 'cuda':
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_gpu_gb = '{:.4f}'.format(peak_bytes / (1024 ** 3))
    else:
        peak_gpu_gb = 'N/A'
    metadata_rows = build_metadata(peak_gpu_gb=peak_gpu_gb,
                                    total_runtime_sec=round(total_runtime_sec, 2))
    save_training_excel(excel_path, valid_iterations, env_steps_at_valid,
                        valid_indist_norm, valid_ood_norm, metadata_rows)
    print("total_time: ", total_runtime_sec)

if __name__ == '__main__':
    main()