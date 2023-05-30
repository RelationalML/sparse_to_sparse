import subprocess


subprocess.call("python cifar_main.py --model Resnet18 --save-model --er-init-density 0.3 --save_file runs-er-strong-tick-anneal-lr-0.1-1 --sparsity 0.1 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 2 |& tee log_signed_const_zero_bias_seed1_sparsity0.1_levels1.txt", shell=True)
