import subprocess

subprocess.call("python cifar_main.py --model Resnet18 --save-model --save_file runs-er-strong-tick-anneal-lr-0.1-3 --sparsity 0.2 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 3", shell=True)

subprocess.call("python cifar_main.py --model Resnet18 --save-model --save_file runs-er-strong-tick-anneal-lr-0.1-3 --sparsity 0.1 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 3", shell=True)

subprocess.call("python cifar_main.py --model Resnet18 --save-model --save_file runs-er-strong-tick-anneal-lr-0.1-4 --sparsity 0.05 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 3", shell=True)

subprocess.call("python cifar_main.py --model Resnet18 --save-model --save_file runs-er-strong-tick-anneal-lr-0.1-4 --sparsity 0.01 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 3", shell=True)

subprocess.call("python cifar_main.py --model Resnet18 --save-model --er-init-density 0.2 --save_file runs-er-strong-tick-anneal-lr-0.1-4 --sparsity 0.05 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 3", shell=True)

subprocess.call("python cifar_main.py --model Resnet18 --save-model --er-init-density 0.3 --save_file runs-er-strong-tick-anneal-lr-0.1-4 --sparsity 0.1 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 3", shell=True)
