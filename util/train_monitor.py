from torch.utils.tensorboard import SummaryWriter

__all__ = ['PythonMonitor', 'TensorBoardMonitor']


class TrainMonitor(object):
    """This is an abstract interface for data loggers

    Train monitors log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """

    def __init__(self):
        pass

    def log_training_progress(self, stats, epoch, completed_batch, total_batch):
        raise NotImplementedError

    def log_activation_stats(self, phase, stat_name, activation_stats, epoch):
        pass

    def log_weights_distribution(self, named_params, steps_completed):
        pass


class PythonMonitor(TrainMonitor):
    def __init__(self, logger):
        super(PythonMonitor, self).__init__()
        self.pylogger = logger

    def log_training_progress(self, stats, epoch, completed_batch, total_batch):
        stats_dict = stats[1]
        if epoch > -1:
            log = 'Epoch: [{}][{:5d}/{:5d}]    '.format(epoch, completed_batch, int(total_batch))
        else:
            log = 'Test: [{:5d}/{:5d}]    '.format(completed_batch, int(total_batch))
        for name, val in stats_dict.items():
            if isinstance(val, int):
                log = log + '{name} {val}    '.format(name=name, val="{:,}".format(val))
            else:
                log = log + '{name} {val:.6f}    '.format(name=name, val=val)
        self.pylogger.info(log)

    # def log_activation_stats(self, phase, stat_name, activation_stats, epoch):
    #     data = []
    #     for layer, statistic in activation_stats.items():
    #         data.append([layer, statistic])
    #     tmp = tabulate.tabulate(data, headers=['Layer', stat_name], tablefmt='psql', floatfmt=".2f")
    #     self.pylogger.info('\n' + tmp)


class TensorBoardMonitor(TrainMonitor):
    def __init__(self, log_dir, logger):
        super(TensorBoardMonitor, self).__init__()
        self.writer = SummaryWriter(log_dir / 'tb_runs')
        logger.info('TensorBoard data directory: %s/tb_runs' % log_dir)

    def log_training_progress(self, stats_dict, epoch, completed_batch, total_batch):
        stats_prefix = stats_dict[0]
        stats_dict = stats_dict[1]
        current_step = epoch * total_batch + completed_batch
        if 'Loss' in stats_dict:
            self.writer.add_scalar(stats_prefix + 'Loss', stats_dict['Loss'], current_step)
        if 'Top1' in stats_dict:
            self.writer.add_scalar(stats_prefix + 'Top1', stats_dict['Top1'], current_step)
        if 'Top5' in stats_dict:
            self.writer.add_scalar(stats_prefix + 'Top5', stats_dict['Top5'], current_step)
