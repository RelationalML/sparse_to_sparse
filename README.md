# Official code for the paper: 'Why Random Pruning Is All We Need to Start Sparse' accepted at ICML 2023 (https://arxiv.org/abs/2210.02412).



## Experiments

### Requirements
See `requirements.txt` for required packages. You can set up a conda environment with this file.

### Random Pruning
Initialize and train an ER network with a specified layerwise sparsity ratio.

Commands:
```
cd STR
python main.py --seed 1 --resnet-type large --config configs/largescale/resnet18-cifar-str-1.yaml --threshold-list  --multigpu 0 --er-sparse-method uniform --er-sparse-init 0.2 --name cifar-imp-er-uniform-0.2 --final_prune_epoch 150 --result-dir results --warmup-epochs 0 --prune-scheduler cosine --expt-setup rml --lr 0.1
```
### ER Dynamic Sparse Training
```
--sparse: Enable sparse mode (remove this if want to train dense model)
--method: Type of sparse training method. Choose from: GraNet, DST, GMP
--sparse-init: Type of sparse initialization. Choose from: ERK, ER_same_params, pyramid, snip, ER_imp, ER_synflow
--init-density (float): Initial density of the sparse model. Default: 0.50
--final-density (float): Target density of the sparse model. Default: 0.05
--model (str): Network architecture. Choose from: vgg16, vgg19, resnet18, resnet50
--data: Dataset. Choose from: cifar10, cifar100, tiny_imagenet
--growth (str): Growth mode. Choose from: random, gradient, momentum
--prune (str): Removing mode. Choose from: magnitude
--prune-rate (float): The pruning rate for growth
--update-frequency (int): number of training iterations between two steps of growth
--flow-preservation: Enables flow correction if True. Default: False
ortho-repair: Enables orthogonal repair in initialization. Default: False 
```
Commands:
```
cd ER_DST
python main.py --sparse --method GraNet --prune-rate 0.5 --optimizer sgd --sparse-init ER_same_params_per_layer --init-density 0.01 --final-density 0.001 --update-frequency 1000 --l2 0.0005 --lr 0.1 --epochs 160 --model vgg16 --data cifar10 --prune magnitude --init-method default --is-prune True --flow-preservation True --ortho-repair False --seed 1
```

### ER Strong Lottery Tickets
Arguments:
```
--model (str): Network architecture. Choose from: VGG16, ResNet18
--er-init-density: Initial density of the sparse model
--sparsity: Final density of the sparse model
--levels: Number of levels to anneal sparsity
```

Commands:
```
cd ER_SLT
python cifar_main.py --model Resnet18 --save-model --er-init-density 0.5 --save_file runs-strong-tick-anneal-lr-0.1-2 --sparsity 0.2 --anneal False --initBias kn-zero-bias --epochs 100 --levels 5 --lr 0.1 --seed 2
```

### Sparse to Sparse (IMP)
Initialize an ER network and perform Iterative Magnitude Pruning starting from a sparse network.
See `run_imp.py` for details to initialize an ER network and perform IMP.

```
cd STR
python run_imp.py
```

### Sparse to Sparse (STR)
Arguments:
```
set: cifar10
name: cifar_0.3_ER_ERK_STR_2

# ===== Learning Rate Policy ======== #
optimizer: sgd
lr: 0.1
lr_policy: cosine_lr
warmup_length: 5

# ===== Network training config ===== #
epochs: 100
weight_decay: 0.001 # Change this according to reported numbers in appendix
momentum: 0.9
batch_size: 256
label_smoothing: 0.1

# ===== Sparsity =========== #
er_sparse_init: 0.3
er_sparse_method: balanced
conv_type: STRConvER
```
Command:


```
cd STR
python main.py --config configs/largescale/resnet18-cifar-str-3.yaml
```

### Tabular Data

See `tabular.ipynb` for an example to run sparse to sparse training on tabular data.

## Acknowledgements
We would like to acknowledge the following code repositories on which our code is based:
- [Random_Pruning](https://github.com/VITA-Group/Random_Pruning)
- [STR](https://github.com/RAIVNLab/STR)
- [GCN](https://github.com/meliketoy/graph-cnn.pytorch)
- [Algorithmic Data](https://github.com/ejmichaud/grokking-squared/blob/main/notebooks/erics-implementation.ipynb)
