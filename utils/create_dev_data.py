import os

from env.case_generator import CaseGenerator

# (num_jobs, num_mas, count) per category. MWR baselines are computed separately.
DEV_SPECS = {
    'indist': [
        (20, 10, 50),
        (30, 10, 30),
        (40, 10, 20),
    ],
    'ood': [
        (50, 10, 10),
        (100, 10, 10),
    ],
}


def generate(folder, num_jobs, num_mas, count):
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)
    case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max,
                          path=folder, flag_same_opes=False, flag_doc=True)
    for idx in range(count):
        case.get_case(idx)


def main():
    for category, specs in DEV_SPECS.items():
        folder = './data_dev/{0}/'.format(category)
        os.makedirs(folder, exist_ok=True)
        for num_jobs, num_mas, count in specs:
            generate(folder, num_jobs, num_mas, count)
            print('Generated {0} instances of {1}x{2} in {3}'.format(count, num_jobs, num_mas, folder))


if __name__ == '__main__':
    main()
