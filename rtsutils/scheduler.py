import numpy as np

class ScheduledOptimizer(object):
    def __init__(self, scheduler, optimizer):
        self.scheduler = scheduler
        self.optimizer = optimizer

    def step(self):
        self.optimizer.step()
        self.scheduler.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def add_param_group(self, *args, **kwargs):
        raise NotImplementedError("need to add initial lr and initial weight decay in add_param_group")
        self.optimizer.add_param_group(*args, **kwargs)

    def state_dict(self,):
        return self.optimizer.state_dict()

    def load_state_dict(self, *args, **kwargs):
        return self.optimizer.load_state_dict(*args, **kwargs)


class CosineAnnealing(object):
    def __init__(self, optimizer, n_updates_per_period, schedule_weight_decay=False):
        if not hasattr(n_updates_per_period, '__len__'):
            n_updates_per_period = [n_updates_per_period]
        assert np.all(np.array(n_updates_per_period) > 0)
        self.update_period_boundaries = np.cumsum(n_updates_per_period)
        self.update_period_boundaries = np.concatenate((
            [0], self.update_period_boundaries))
        self.schedule_weight_decay = schedule_weight_decay
        self.initial_lrs = list(map(
            lambda group: group['lr'], optimizer.param_groups))
        self.initial_weight_decays = list(map(
            lambda group: group['weight_decay'], optimizer.param_groups))
        self.optimizer = optimizer
        self.i_update = 0

    def step(self):
        self.i_update += 1
        for group, initial_lr, initial_wd in zip(
                self.optimizer.param_groups,
                self.initial_lrs,
                self.initial_weight_decays):
            group['lr'] = self.get_lr(initial_lr, self.i_update)
            if self.schedule_weight_decay:
                group['weight_decay'] = self.get_weight_decay(
                    initial_wd, self.i_update)

    def get_lr(self, initial_val, i_update):
        i_update = i_update % self.update_period_boundaries[-1]
        assert i_update < self.update_period_boundaries[-1], (
            "More updates ({:d}) than expected ({:d})".format(
                i_update, self.update_period_boundaries[-1] - 1))
        i_end_period = np.searchsorted(self.update_period_boundaries,
                                       i_update, side='right')
        assert i_end_period > 0
        i_start_update = self.update_period_boundaries[i_end_period - 1]
        i_end_update = self.update_period_boundaries[i_end_period]
        i_update = i_update - i_start_update
        assert i_update >= 0
        n_updates_this_period = i_end_update - i_start_update
        fraction_period = i_update / np.float64(n_updates_this_period)
        return initial_val * (0.5 * np.cos(np.pi * fraction_period) + 0.5)

    def get_weight_decay(self, initial_val, i_update):
        return self.get_lr(initial_val, i_update)
