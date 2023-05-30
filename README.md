# How Erdös and Rényi Win the Lottery

## Requirements
- Python 3.6
- PyTorch v1.5.1
- CUDA v10.2

## Experiments


### ER Weak Lottery Tickets
Arguments:
```
--sparse: Enable sparse mode
--fix: Fix the sparse pattern during training
--sparse-init: Type of sparse initialization. Choose from: ERK, ER_same_params, pyramid, snip, ER_imp, ER_synflow
--model (str): Network architecture. Choose from: vgg16, vgg19, resnet18, resnet20, resnet50
--data: Dataset. Choose from: cifar10, cifar100, tiny_imagenet
--density (float): Density level. Default: 0.05
```

Commands:
```
cd ER_WLT
python main.py --sparse --seed 1 --sparse_init ER_same_params_per_layer --fix --lr 0.1 --density 0.01 --model resnet50 --data cifar100 --epoch 160
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

### ER with STR
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

## Pre-trained Models
The following pre-trained models (based on the repository [Synaptic-Flow](https://github.com/ganguli-lab/Synaptic-Flow)) should be downloaded for running experiments with `ER_imp` and `ER_synflow` masks:

| Dataset/Method | IMP | SynFlow |
|---|---|---|
| CIFAR10 | [models](https://drive.google.com/drive/folders/1aZqABifecBbniC9sU4s8UUbRr5m00GPm?usp=sharing) | [models](https://drive.google.com/drive/folders/1MYAO7stAgGG6nrirPUXz0pBk65duLdeL?usp=sharing) |
| CIFAR100 | [models](https://drive.google.com/drive/folders/1ChFGpFymGP23ucW3DlgLL9ONidyYv21U?usp=sharing) | [models](https://drive.google.com/drive/folders/1Lwk5rrFdgh0vFFYGprcVXVJvQLW_kasf?usp=sharing) |


## Contributions
We have added two new random masks to pre-existing code, namely: Uniform (`ER_same_params`) and Pyramid (`pyramid`). We extended ER mask to `ER_imp` and `ER_synflow`. We have also introduced the flow correction (`flow-preservation`) functionality. 

## Acknowledgements
We would like to acknowledge the following code repositories on which our code is based:
- [Random_Pruning](https://github.com/VITA-Group/Random_Pruning)
- [Synaptic-Flow](https://github.com/ganguli-lab/Synaptic-Flow)
