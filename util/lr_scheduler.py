import math


def lr_scheduler(optimizer, mode, batch_size=None, num_samples=None, update_per_batch=False, **kwargs):
    # variables batch_size & num_samples are only used when the learning rate updated every epoch
    if update_per_batch:
        assert isinstance(batch_size, int) and isinstance(num_samples, int)

    if mode == 'fixed':
        scheduler = FixedLr
    elif mode == 'step':
        scheduler = StepLr
    elif mode == 'multi_step':
        scheduler = MultiStepLr
    elif mode == 'exp':
        scheduler = ExponentialLr
    elif mode == 'cos':
        scheduler = CosineLr
    elif mode == 'cos_warm_restarts':
        scheduler = CosineWarmRestartsLr
    else:
        raise ValueError('LR scheduler `%s` is not supported', mode)

    return scheduler(optimizer=optimizer, batch_size=batch_size, num_samples=num_samples,
                     update_per_batch=update_per_batch, **kwargs)


class LrScheduler:
    def __init__(self, optimizer, batch_size, num_samples, update_per_batch):
        self.optimizer = optimizer
        self.current_lr = self.get_lr()
        self.base_lr = self.get_lr()
        self.num_groups = len(self.base_lr)

        self.batch_size = batch_size
        self.num_samples = num_samples
        self.update_per_batch = update_per_batch

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]

    def set_lr(self, lr):
        for i in range(self.num_groups):
            self.current_lr[i] = lr[i]
            self.optimizer.param_groups[i]['lr'] = lr[i]

    def step(self, epoch, batch):
        raise NotImplementedError

    def __str__(self):
        s = '`%s`' % self.__class__.__name__
        s += '\n    Update per batch: %s' % self.update_per_batch
        for i in range(self.num_groups):
            s += '\n             Group %d: %g' % (i, self.current_lr[i])
        return s


class FixedLr(LrScheduler):
    def step(self, epoch, batch):
        pass


class LambdaLr(LrScheduler):
    def __init__(self, lr_lambda, **kwargs):
        super(LambdaLr, self).__init__(**kwargs)
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * self.num_groups
        else:
            if len(lr_lambda) != self.num_groups:
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    self.num_groups, len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)

    def step(self, epoch, batch):
        if self.update_per_batch:
            epoch = epoch + batch * self.batch_size / self.num_samples
        for i in range(self.num_groups):
            func = self.lr_lambdas[i]
            self.current_lr[i] = func(epoch) * self.base_lr[i]
        self.set_lr(self.current_lr)


class StepLr(LrScheduler):
    def __init__(self, step_size=30, gamma=0.1, **kwargs):
        super(StepLr, self).__init__(**kwargs)
        self.step_size = step_size
        self.gamma = gamma

    def step(self, epoch, batch):
        for i in range(self.num_groups):
            self.current_lr[i] = self.base_lr[i] * (self.gamma ** (epoch // self.step_size))
        self.set_lr(self.current_lr)


class MultiStepLr(LrScheduler):
    def __init__(self, milestones=[30, ], gamma=0.1, **kwargs):
        super(MultiStepLr, self).__init__(**kwargs)
        self.milestones = milestones
        self.gamma = gamma

    def step(self, epoch, batch):
        n = sum([1 for m in self.milestones if m <= epoch])
        scale = self.gamma ** n
        for i in range(self.num_groups):
            self.current_lr[i] = self.base_lr[i] * scale
        self.set_lr(self.current_lr)


class ExponentialLr(LrScheduler):
    def __init__(self, gamma=0.95, **kwargs):
        super(ExponentialLr, self).__init__(**kwargs)
        self.gamma = gamma

    def step(self, epoch, batch):
        if self.update_per_batch:
            epoch = epoch + batch * self.batch_size / self.num_samples
        for i in range(self.num_groups):
            self.current_lr[i] = self.base_lr[i] * (self.gamma ** epoch)


class CosineLr(LrScheduler):
    def __init__(self, lr_min=0., cycle=90, **kwargs):
        super(CosineLr, self).__init__(**kwargs)
        self.min_lr = lr_min
        self.cycle = cycle

    def step(self, epoch, batch):
        if self.update_per_batch:
            epoch = epoch + batch * self.batch_size / self.num_samples
        if epoch > self.cycle:
            epoch = self.cycle
        for i in range(self.num_groups):
            self.current_lr[i] = self.min_lr + 0.5 * (self.base_lr[i] - self.min_lr) \
                                 * (1 + math.cos(math.pi * epoch / self.cycle))
        self.set_lr(self.current_lr)


class CosineWarmRestartsLr(LrScheduler):
    def __init__(self, lr_min=0., cycle=5, cycle_scale=2., amp_scale=0.5, **kwargs):
        super(CosineWarmRestartsLr, self).__init__(**kwargs)
        self.min_lr = lr_min
        self.cycle = cycle
        self.cycle_scale = cycle_scale
        self.amp_scale = amp_scale

    def step(self, epoch, batch):
        if self.update_per_batch:
            epoch = epoch + batch * self.batch_size / self.num_samples

        curr_cycle = self.cycle
        curr_amp = 1.
        while epoch >= curr_cycle:
            epoch = epoch - curr_cycle
            curr_cycle *= self.cycle_scale
            curr_amp *= self.amp_scale

        for i in range(self.num_groups):
            self.current_lr[i] = self.min_lr + 0.5 * curr_amp * (self.base_lr[i] - self.min_lr) \
                                 * (1 + math.cos(math.pi * epoch / curr_cycle))
        self.set_lr(self.current_lr)
