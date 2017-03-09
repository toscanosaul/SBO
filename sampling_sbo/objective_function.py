import numpy as np
import multiprocessing as mp
from joblib import(
    Parallel,
    delayed,
)

MAX_RANDOM_SEED = 1000000000

class Objective(object):

    def __init__(self, f, **options):
        self.function = f
        self.nEvals = int(options.get('nEvals', 1))
        self.dim_x = int(options.get('dim_x'))
        self.dim_w = int(options.get('dim_w', 0))
        self.nCores = mp.cpu_count()
        self.domain = options.get('domain')
        self.type_domain = options.get('type_domain', None)

        self._setup()

    def _setup(self):
        if self.type_domain is None:
            self.type_domain = (self.dim_x + self.dim_w) * ['real']

    def evaluate_function(self, xw):
        x = xw[0:self.dim_x]
        w = xw[self.dim_x:self.dim_x+self.dim_w]

        if self.nEvals == 1:
            samples = self.function(x, w)
            return samples, 0
        else:
            random_seeds = np.random.randint(1, MAX_RANDOM_SEED+1,self.nEvals)
            samples = Parallel(n_jobs=self.nCores)(
                delayed(self.function)(
                    x=x,
                    w=w,
                    random_seed=random_seeds[i]
                ) for i in range(self.nEvals)
            )

            sample_results = np.zeros(self.nEvals)
            for i in range(self.nEvals):
                sample_results[i] = samples[i]

            return np.mean(sample_results), np.std(sample_results)

    def evaluate_function_several_points(self, XW):

        n_points = XW.shape[0]

        random_seeds = np.random.randint(1, MAX_RANDOM_SEED + 1, self.nEvals * n_points)
        samples = Parallel(n_jobs=self.nCores)(
            delayed(self.function)(
                x=XW[j,0:self.dim_x],
                w=XW[j,self.dim_x:self.dim_x+self.dim_w],
                random_seed=random_seeds[j*self.nEvals + i]
            ) for j in range(n_points) for i in range(self.nEvals)
        )

        evaluations = np.zeros([n_points, 2])

        for i in range(n_points):
            samples_results = np.zeros(self.nEvals)
            for j in range(self.nEvals):
                samples_results[j] = samples[i*self.nEvals + j]
            evaluations[i, 0] = np.mean(samples_results)
            evaluations[i, 1] = np.std(samples_results)

        return evaluations

    def generate_random_points_in_domain(self, n, only_x=False):
        if only_x:
            points = np.zeros([n, self.dim_x])
        else:
            points = np.zeros([n, self.dim_x+self.dim_w])

        for i in range(self.dim_x):
            if self.type_domain[i] == 'real':
                points[:,i] = np.random.uniform(
                    self.domain[i]['lower'],
                    self.domain[i]['upper'],
                    n
                )
            else:
                points[:, i] = np.random.randint(
                    self.domain[i]['lower'],
                    self.domain[i]['upper'],
                    n
                )
        if only_x:
            return points

        for j in range(self.dim_w):
            if self.type_domain[self.dim_x+j] == 'real':
                points[:, j+self.dim_x] = np.random.uniform(
                    self.domain[j+self.dim_x]['lower'],
                    self.domain[j+self.dim_x]['upper'],
                    n
                )
            else:
                points[:, j+self.dim_x] = np.random.randint(
                    self.domain[j+self.dim_x]['lower'],
                    self.domain[j+self.dim_x]['upper'],
                    n
                )

        return points

    def generate_training_points(self, n):
        XW = self.generate_random_points_in_domain(n)
        evaluations = self.evaluate_function_several_points(XW)

        return XW, evaluations



