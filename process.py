import logging
import math
import operator
import time

import torch as t

from util import AverageMeter

__all__ = ['train', 'validate', 'PerformanceScoreboard']

logger = logging.getLogger()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with t.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device.type)
        targets = targets.to(args.device.type)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        if lr_scheduler is not None:
            lr_scheduler.step(epoch=epoch, batch=batch_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.log.print_freq == 0:
            for m in monitors:
                m.update(epoch, batch_idx + 1, steps_per_epoch, 'Training', {
                    'Loss': losses,
                    'Top1': top1,
                    'Top5': top5,
                    'BatchTime': batch_time,
                    'LR': optimizer.param_groups[0]['lr']
                })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n',
                top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


def validate(data_loader, model, criterion, epoch, monitors, args):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(args.device.type)
            targets = targets.to(args.device.type)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.log.print_freq == 0:
                for m in monitors:
                    m.update(epoch, batch_idx + 1, steps_per_epoch, 'Validation', {
                        'Loss': losses,
                        'Top1': top1,
                        'Top5': top5,
                        'BatchTime': batch_time
                    })

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores):
        self.board = list()
        self.num_best_scores = num_best_scores

    def update(self, top1, top5, epoch):
        """ Update the list of top training scores achieved so far, and log the best scores so far"""
        self.board.append({'top1': top1, 'top5': top5, 'epoch': epoch})

        # Keep scoreboard sorted from best to worst, and sort by top1, top5 and epoch
        curr_len = min(self.num_best_scores, len(self.board))
        self.board = sorted(self.board,
                            key=operator.itemgetter('top1', 'top5', 'epoch'),
                            reverse=True)[0:curr_len]
        for idx in range(curr_len):
            score = self.board[idx]
            logger.info('Scoreboard best %d ==> Epoch [%d][Top1: %.3f   Top5: %.3f]',
                        idx + 1, score['epoch'], score['top1'], score['top5'])

    def is_best(self, epoch):
        return self.board[0]['epoch'] == epoch
