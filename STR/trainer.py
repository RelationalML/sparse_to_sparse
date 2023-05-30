import time
import torch
import tqdm
import numpy as np

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate"]


def hessian_trace(train_loader, model, criterion, optimizer, args):
    model.train()
    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        break
    if args.hessian:
        from pyhessian import hessian
        if args.gpu is not None:
            images = images.cuda()

        target = target.cuda().long()

        hessian_comp = hessian(model, criterion, data=(images, target), cuda=True)
        trace = hessian_comp.trace()
        trace = np.array(trace).mean()
        print('trace of hessian: ', trace)
    return trace


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True).long()

        # compute output
        output = model(images)
        loss = criterion(output, target.view(-1))

        # measure accuracy and record loss
        if args.set == 'heart':
            acc1 = accuracy(output, target, topk=[1])[0]
            acc5 = torch.tensor(0)
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

            loss = criterion(output, target.view(-1))

            # measure accuracy and record loss
            if args.set == 'heart':
                acc1 = accuracy(output, target, topk=[1])[0]
                acc5 = torch.tensor(0)
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

    return top1.avg, top5.avg

def validate_loss(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter("Time", ":6.3f", write_val=False)
    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        val_loss_avg = 0
        cnt = 0
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True).long()

            # compute output
            output = model(images)

            loss = criterion(output, target.view(-1))

            # measure accuracy and record loss
            if args.set == 'heart':
                acc1 = accuracy(output, target, topk=[1])[0]
                acc5 = torch.tensor(0)
            else:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            val_loss_avg += loss
            cnt += 1

        progress.display(len(val_loader))

    return top1.avg, top5.avg, val_loss_avg / cnt


def get_preds(train_loader_pred, val_loader, model, criterion, optimizer, epoch, args, writer):
    
    model.eval()

    batch_size = train_loader_pred.batch_size
    num_batches = len(train_loader_pred)
    end = time.time()
    train_pred = []
    train_true = []

    val_pred = []
    val_true = []
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader_pred), ascii=True, total=len(train_loader_pred)):
        # measure data loading time
        
        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True).long()

        # compute output
        output = model(images)
        preds = torch.argmax(output, dim=1)
        train_pred.extend(preds)
        train_true.extend(target)

    for i, (images, target) in tqdm.tqdm(
        enumerate(val_loader), ascii=True, total=len(val_loader)):

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        target = target.cuda(args.gpu, non_blocking=True).long()
        
        # compute output
        output = model(images)
        preds = torch.argmax(output, dim=1)
        val_pred.extend(preds)
        val_true.extend(target)
    
    return train_pred, train_true, val_pred, val_true
    