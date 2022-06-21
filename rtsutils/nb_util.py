import contextlib
from contextlib import contextmanager

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
import time

from .plot import display_close


class Results(object):
    def __init__(self, decay=0.95):
        self.out_print = widgets.Output(layout={'border': '1px solid black'})
        self.out_plot = widgets.Output(layout={'border': '1px solid black'})
        display(self.out_print)
        display(self.out_plot)
        self.metrics_df = pd.DataFrame()
        self.metrics_average = None
        self.fig = None
        self.output_areas = {}
        self.decay = decay
        self.last_print_time = -1

    def collect(self, **metrics):
        if self.metrics_average is None:
            self.metrics_average = metrics
        else:
            for key in self.metrics_average:
                self.metrics_average[key] = (
                        self.metrics_average[key] * self.decay +
                        metrics[key] * (1-self.decay))
        self.metrics_df = self.metrics_df.append(metrics, ignore_index=True)

    def plot_df(self):
        self.out_plot.clear_output(wait=True)
        with self.out_plot:
            n_plots = len(self.metrics_df.columns)
            n_rows = int(np.ceil(n_plots / 3.0))
            fig, axes = plt.subplots(n_rows, 3,
                                     figsize=(14, 3 * n_rows))
            # loop over metrics average as keys are in correct oder in that dict
            for i_plot, metric_name in enumerate(self.metrics_average):
                axes.flatten()[i_plot].plot(self.metrics_df.loc[:, metric_name])
                axes.flatten()[i_plot].set_title(metric_name)
            display_close(fig)

    def print(self):
        # only print every sec
        if (time.time() - self.last_print_time) > 1:
            self.out_print.clear_output(wait=True)
            with self.out_print:
                for key, val in self.metrics_average.items():
                    print(f"{key:20s} {val:.2f}")
            self.last_print_time = time.time()

    @contextmanager
    def output_area(self, name):
        if name not in self.output_areas:
            out = widgets.Output(layout={'border': '1px solid black'})
            display(out)
            self.output_areas[name] = out
        self.output_areas[name].clear_output(wait=True)
        with self.output_areas[name] as c:
            try:
                yield [c]
            finally:
                pass

class TerminalResults(object):
    def __init__(self, tqdm_obj, decay):
        self.metrics_df = pd.DataFrame()
        self.metrics_average = None
        self.fig = None
        self.decay = decay
        self.tqdm_obj = tqdm_obj

    def collect(self, **metrics):
        if self.metrics_average is None:
            self.metrics_average = metrics
        else:
            for key in self.metrics_average:
                self.metrics_average[key] = (
                        self.metrics_average[key] * self.decay +
                        metrics[key] * (1-self.decay))
        self.metrics_df = self.metrics_df.append(metrics, ignore_index=True)

    def print(self):
        self.tqdm_obj.set_postfix(self.metrics_average.items())

    def plot_df(self):
        return

    @contextmanager
    def output_area(self, name):
        yield contextlib.nullcontext()

class NoOpResults(object):
    def __init__(self, decay):
        return

    def collect(self, **metrics):
        return

    def print(self):
        return

    def plot_df(self):
        return

    @contextmanager
    def output_area(self, name):
        yield contextlib.nullcontext()
