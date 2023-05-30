import subprocess
import numpy as np


seeds = np.arange(15, 17)
for seed in seeds:
    for samples in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        for density in [0.5, 0.3, 0.1]:
            dim = 10
            noise = 0.5
            prune_iters = 3
            epochs = 50000
            subprocess.call('python finetune_theory_example.py --result-dir results-finetune-noise-fixed-0.5-2 --reset-weights --name imp --dim '+str(dim)+' --num-samples '+str(samples)+' --prune-iters '+str(prune_iters)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --density '+str(density), shell=True)
            subprocess.call('python finetune_theory_example.py --result-dir results-finetune-noise-fixed-0.5-2 --name finetune --dim '+str(dim)+' --num-samples '+str(samples)+' --prune-iters '+str(prune_iters)+' --epochs '+str(epochs)+' --seed '+str(seed)+' --noise '+str(noise)+' --density '+str(density), shell=True)
