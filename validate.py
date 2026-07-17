import PPO_model
from env.fjsp_env import FJSPEnv
from env.load_data import nums_detec
import torch
import time
import os
import copy
import csv


def load_mwr_baselines(csv_path):
    '''
    Reads a CSV file with columns 'instance_name' and 'makespan' into a dict.
    '''
    mwr_baselines = {}
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mwr_baselines[row['instance_name']] = float(row['makespan'])
    return mwr_baselines


def group_files_by_size(folder_path):
    '''
    Groups .fjs files in folder_path by (num_jobs, num_mas), read from each
    file's header line (not the filename).
    '''
    groups = {}
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith('.fjs'):
            continue
        filepath = os.path.join(folder_path, filename)
        with open(filepath) as f:
            num_jobs, num_mas, _ = nums_detec(f.readlines())
        groups.setdefault((num_jobs, num_mas), []).append(filepath)
    return groups


def get_validate_envs(env_paras):
    '''
    Generate and return the validation environments (in-distribution and out-of-distribution),
    grouped by instance size, along with their MWR baselines.
    '''
    valid_envs = {}
    for category in ('indist', 'ood'):
        folder = './data_dev/{0}/'.format(category)
        mwr_baselines = load_mwr_baselines(os.path.join(folder, 'mwr_baselines.csv'))
        groups = group_files_by_size(folder)

        category_envs = []
        for (nj, nm), filepaths in sorted(groups.items()):
            env_p = copy.deepcopy(env_paras)
            env_p["num_jobs"] = nj
            env_p["num_mas"] = nm
            env_p["batch_size"] = len(filepaths)
            env = FJSPEnv(case=filepaths, env_paras=env_p, data_source='file')

            filenames = [os.path.basename(fp) for fp in filepaths]
            mwr_dict = {filename: mwr_baselines[filename] for filename in filenames}
            size_label = f"{nj}x{nm}"

            category_envs.append((env, mwr_dict, size_label, filenames))

        valid_envs[category] = category_envs

    return valid_envs


def validate(valid_envs, model_policy):
    '''
    Validate the policy during training on the in-distribution and out-of-distribution
    validation sets. Makespans are normalized by MWR baselines and averaged per size
    group (equal weight per size), then averaged across size groups per category.
    '''
    start = time.time()
    category_scores = {}

    for category in ('indist', 'ood'):
        size_means = []
        for env, mwr_dict, size_label, filenames in valid_envs[category]:
            memory = PPO_model.Memory()
            state = env.state
            done = False
            dones = env.done_batch
            while ~done:
                with torch.no_grad():
                    actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
                state, rewards, dones = env.step(actions)
                done = dones.all()

            gantt_result = env.validate_gantt()[0]
            if not gantt_result:
                print("Scheduling Error！！！！！！")

            makespan_batch = env.makespan_batch.to('cpu')
            mwr_tensor = torch.tensor([mwr_dict[filename] for filename in filenames])
            normalized = makespan_batch / mwr_tensor
            size_mean = normalized.mean().item()
            size_means.append(size_mean)

            env.reset()
            print(f'  {category}/{size_label}: norm={size_mean:.4f} (n={len(filenames)})')

        category_scores[category] = sum(size_means) / len(size_means)

    indist_score = category_scores['indist']
    ood_score = category_scores['ood']
    print(f'  indist_norm={indist_score:.4f}, ood_norm={ood_score:.4f}')
    print('validating time: ', time.time() - start, '\n')
    return indist_score, ood_score
