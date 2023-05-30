import numpy as np

__all__ = ["multistep_lr", "cosine_lr", "constant_lr", "efficientnet_lr", "get_policy"]


def get_policy(name):
    if name is None:
        return constant_lr

    out_dict = {
        "constant_lr": constant_lr,
        "cosine_lr": cosine_lr,
        "efficientnet_lr": efficientnet_lr,
        "multistep_lr": multistep_lr,
    }

    return out_dict[name]


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def constant_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch):
        # if epoch < args.warmup_length:
        #     lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        # else:
        lr = args.lr
        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def warmup_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch):
        lr = _warmup_lr(args.lr, args.warmup_epochs, epoch)

        assign_learning_rate(optimizer, lr)

    return _lr_adjuster

# define scheduler
def cosine_lr(optimizer, total_epochs, args, **kwargs):
    def _lr_adjuster(epoch):
    
        e = epoch
        es = total_epochs
        lr = args.lr_min + 0.5 * (1 + np.cos(np.pi * e / es)) * (args.lr - args.lr_min)

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster

def efficientnet_lr(optimizer, args, **kwargs):
    def _lr_adjuster(epoch, iteration):
        if epoch < args.warmup_length:
            lr = _warmup_lr(args.lr, args.warmup_length, epoch)
        else:
            lr = args.lr * (0.97 ** (epoch / 2.4))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def multistep_lr(optimizer, args, **kwargs):
    """Sets the learning rate to the initial LR decayed by 10 every 60 epochs"""

    def _lr_adjuster(epoch):
        lr = args.lr * (0.1 ** (epoch // 60))

        assign_learning_rate(optimizer, lr)

        return lr

    return _lr_adjuster


def _warmup_lr(base_lr, warmup_length, epoch):
    return base_lr * (epoch + 1) / warmup_length
