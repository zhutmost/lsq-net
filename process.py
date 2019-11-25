import torch as t
import logging
import time
from collections import OrderedDict
import math
import operator


logger = logging.getLogger()


class AverageMeter(object):
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, monitors, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total_sample = len(train_loader.sampler)
    batch_size = train_loader.batch_size
    steps_per_epoch = math.ceil(total_sample / batch_size)
    logger.info('Training: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.train()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        top5.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step(epoch)

        batch_time.update(time.time() - end_time)
        end_time = time.time()

        if (batch_idx + 1) % args.print_freq == 0:
            status_dict = OrderedDict()
            status_dict['Loss'] = losses.avg
            status_dict['Top1'] = top1.avg
            status_dict['Top5'] = top5.avg
            status_dict['Time'] = batch_time.avg
            status = ('Performance/Training/', status_dict)
            for m in monitors:
                m.log_training_progress(status, epoch, (batch_idx + 1), steps_per_epoch, args.print_freq)

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f    LearningRate: %.6f\n',
                top1.avg, top5.avg, losses.avg, lr_scheduler.get_lr()[0])
    return top1.avg, top5.avg, losses.avg


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def validate(data_loader, model, criterion, epoch, monitors, args):
    losses = AverageMeter()
    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    total_sample = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    total_step = total_sample / batch_size

    logger.info('Validation: %d samples (%d per mini-batch)', total_sample, batch_size)

    model.eval()
    end_time = time.time()
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        with t.no_grad():
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            top5.update(acc5.item(), inputs.size(0))
            batch_time.update(time.time() - end_time)
            end_time = time.time()

            if (batch_idx + 1) % args.print_freq == 0:
                status_dict = OrderedDict()
                status_dict['Loss'] = losses.avg
                status_dict['Top1'] = top1.avg
                status_dict['Top5'] = top5.avg
                status_dict['Time'] = batch_time.avg
                status = ('Performance/Validation/', status_dict)
                for m in monitors:
                    m.log_training_progress(status, epoch, (batch_idx + 1), total_step, args.print_freq)

    logger.info('==> Top1: %.3f    Top5: %.3f    Loss: %.3f\n', top1.avg, top5.avg, losses.avg)
    return top1.avg, top5.avg, losses.avg


class MutableNamedTuple(dict):
    def __init__(self, init_dict):
        for k, v in init_dict.items():
            self[k] = v

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, val):
        if key in self.__dict__:
            self.__dict__[key] = val
        else:
            self[key] = val


def update_training_scoreboard(perf_scoreboard, model, top1, top5, epoch, num_best_scores):
    """ Update the list of top training scores achieved so far, and log the best scores so far"""

    perf_scoreboard.append(MutableNamedTuple({'top1': top1, 'top5': top5, 'epoch': epoch}))
    # Keep perf_scores_history sorted from best to worst
    # Sort by top1, top5 and epoch
    perf_scoreboard.sort(key=operator.attrgetter('top1', 'top5', 'epoch'), reverse=True)
    for score in perf_scoreboard[:num_best_scores]:
        logger.info('==> Best @ Epoch [%d][Top1: %.3f   Top5: %.3f]',
                    score.epoch, score.top1, score.top5)
