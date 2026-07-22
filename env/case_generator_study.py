import math
import random
import time

import numpy as np
from scipy.stats import norm


class CaseGenerator:
    '''
    FJSP instance generator, extended for the Hurink-comparison study with two
    additional controls not present in the original generator:
      - target_flexibility: desired average number of eligible machines per
        operation (instead of a uniform draw between 1 and num_mas).
      - job_corr: target Pearson correlation (0..1) between the mean
        processing times of operations belonging to the same job, generated
        via a one-factor Gaussian model (X_ope = sqrt(rho)*Z_job + sqrt(1-rho)*Z_ope).
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='../data/',
                 flag_same_opes=True, flag_doc=False, target_flexibility=None, job_corr=0.0):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_mas
        self.target_flexibility = target_flexibility  # Desired avg. machines/operation, if set
        self.job_corr = job_corr  # Target Pearson correlation between ope. mean times of the same job
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = 20
        self.proctime_dev = 0.2

    def get_case(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        if self.target_flexibility is None:
            self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        else:
            # 1 + Binomial(num_mas-1, p) has mean = 1 + (num_mas-1)*p = target_flexibility
            p = (self.target_flexibility - 1) / (self.num_mas - 1)
            self.nums_option = [1 + int(np.random.binomial(self.num_mas - 1, p)) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas+1), val))
        self.proc_time = []
        self.proc_times_mean = [0] * self.num_opes
        for j in range(self.num_jobs):
            z_job = random.gauss(0, 1)
            for k in range(self.nums_ope[j]):
                z_ope = random.gauss(0, 1)
                x = math.sqrt(self.job_corr) * z_job + math.sqrt(1 - self.job_corr) * z_ope
                u = norm.cdf(x)  # maps the correlated normal draw to (0, 1)
                mean_val = self.proctime_per_ope_min + u * (self.proctime_per_ope_max - self.proctime_per_ope_min)
                mean_val = int(round(mean_val))
                mean_val = max(self.proctime_per_ope_min, min(self.proctime_per_ope_max, mean_val))
                self.proc_times_mean[self.num_ope_biass[j] + k] = mean_val
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min,round(self.proc_times_mean[i]*(1-self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max,round(self.proc_times_mean[i]*(1+self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1),3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs
