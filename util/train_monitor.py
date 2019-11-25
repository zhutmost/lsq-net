__all__ = ['PythonMonitor']


class TrainMonitor(object):
    """This is an abstract interface for data loggers

    Train monitors log the progress of the training process to some backend.
    This backend can be a file, a web service, or some other means to collect and/or
    display the training
    """
    def __init__(self):
        pass

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        raise NotImplementedError

    def log_activation_stats(self, phase, stat_name, activation_stats, epoch):
        pass

    def log_weights_distribution(self, named_params, steps_completed):
        pass


class PythonMonitor(TrainMonitor):
    def __init__(self, logger):
        super(PythonMonitor, self).__init__()
        self.pylogger = logger

    def log_training_progress(self, stats_dict, epoch, completed, total, freq):
        stats_dict = stats_dict[1]
        if epoch > -1:
            log = 'Epoch: [{}][{:5d}/{:5d}]    '.format(epoch, completed, int(total))
        else:
            log = 'Test: [{:5d}/{:5d}]    '.format(completed, int(total))
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
