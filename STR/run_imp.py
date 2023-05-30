import subprocess
import numpy as np

prune_frac = 0.8
curr_density = 0.3
for seed in [1]:
    for density in [0.01]:
        f = density
        L = int((np.log(density) - np.log(curr_density))/ np.log(prune_frac))
        l = np.arange(1, L + 1)
        y = curr_density * prune_frac ** l
        y_str = "".join(str(elem)+ ' ' for elem in y)   
        subprocess.call('python main.py --seed '+ str(seed) +' --resnet-type large --config configs/largescale/resnet18-cifar-str-1.yaml --threshold-list '+y_str+' --multigpu 0 --pruner mag --reset-weights --er-sparse-method balanced --er-sparse-init 0.3 --name cifar-imp-er-balanced-0.3-'+str(density)+'  --final_prune_epoch 150 --result-dir results --warmup-epochs 0 --prune-scheduler cosine --lr 0.1', shell=True)
