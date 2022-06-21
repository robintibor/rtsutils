import torch as th
import logging

log = logging.getLogger(__name__)


def step_and_clear_gradients(optimizer):
    optimizer.step()
    optimizer.zero_grad()


def check_gradients_clear(optimizer):
    for g in optimizer.param_groups:
        for p in g['params']:
            assert p.grad is None or th.all(p.grad.data == 0).item(), (
                "Gradient not none or zero!")


def grads_all_finite(optimizer):
    for g in optimizer.param_groups:
        for p in g['params']:
            if p.grad is None:
                log.warning("Gradient was none on check of finite grads")
            elif not th.all(th.isfinite(p.grad)).item():
                return False
    return True

