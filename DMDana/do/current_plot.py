#! /usr/bin/env python
"""
This script is used for plotting current images.
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as sgl
from scipy.signal import savgol_filter

from ..lib import constant as const
from .config import DMDana_ini_config_setting_class, config_current


class plot_current_plot:
    def __init__(self, config: config_current):
        self.config = config
        self.configsetting = config.configsetting
        self.fig, self.ax = None, None
        self.timedata = None
        total_time = self.configsetting.t_max - self.configsetting.t_min
        self.mintime_plot = self.configsetting.t_min - 0.05 * total_time
        self.maxtime_plot = (np.max(self.config.jx_data[:,0]) / const.fs + 0.05 * total_time
                             if self.configsetting.t_max == -1 else
                             self.configsetting.t_max + 0.05 * total_time)

    def plot(self):
        self._setup_figure()
        self.timedata = self.config.jx_data[:, 0] / const.fs
        self._check_data()
        
        if self.config.configsetting.only_jtot:
            self._plot_tot()
        else:
            self._plot_tot_diag_offdiag()
            
        self.fig.tight_layout()
        self._save_fig()
        plt.close(self.fig)

    def _check_data(self):
        # Ensure data arrays are not empty
        assert len(self.config.jx_data) > 0, "jx_data is empty, please check the data file."
        assert len(self.config.jy_data) > 0, "jy_data is empty, please check the data file."
        assert len(self.config.jz_data) > 0, "jz_data is empty, please check the data file."

    def _setup_figure(self):
        # Setup figure layout based on the plotting mode
        if self.config.configsetting.only_jtot:
            self.fig, self.ax = plt.subplots(1, 3, figsize=(10, 6), dpi=200, sharex=True)
        else:
            self.fig, self.ax = plt.subplots(3, 3, figsize=(10, 6), dpi=200, sharex=True)

    def _plot_tot_diag_offdiag(self):
        self.fig.suptitle('Current' + self.config.light_label)
        
        for jdata, jdirection, j in zip([self.config.jx_data, self.config.jy_data, self.config.jz_data], 
                                        ['x', 'y', 'z'], range(3)):
            for i in range(3):
                self._format_ax(self.ax[i][j])
            self._set_labels(j)
            self._plot_data(self.ax[0][j], jdata[:, 1])
            self._plot_data(self.ax[1][j], jdata[:, 2]) 
            self._plot_data(self.ax[2][j], jdata[:, 3])

    def _plot_tot(self):
        self.fig.suptitle('Current' + self.config.light_label)
        
        for jdata, jdirection, j in zip([self.config.jx_data, self.config.jy_data, self.config.jz_data], 
                                        ['x', 'y', 'z'], range(3)):
            self._format_ax(self.ax[j])
            self.ax[0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
            self.ax[j].set_title(jdirection)
            self.ax[j].set_xlabel('t (fs)')
            self._plot_data(self.ax[j], jdata[:, 1])
            self.ax[j].set_xlim(self.mintime_plot, self.maxtime_plot)

    def _save_fig(self):
        # Save the figure with a filename that reflects the smoothing settings
        smooth_str = 'off' if not self.configsetting.smooth_on else f'on_{self.configsetting.smooth_method}_smoothtimes_{self.configsetting.smooth_times}'
        if self.configsetting.smooth_method == 'flattop':
            smooth_str += f'_windowlen_{self.configsetting.smooth_windowlen}'
        self.fig.savefig(f"j_smooth_{smooth_str}_{self.config.elec_or_hole_this}.png")

    def _format_ax(self, ax):
        # Format the axis for scientific notation
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.yaxis.major.formatter._useMathText = True
        ax.set_xlim(self.mintime_plot, self.maxtime_plot)

    def _set_labels(self, j):
        # Set labels for the axes
        self.ax[0][0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
        self.ax[1][0].set_ylabel('$j^{diag}(t)$ A/cm$^2$')
        self.ax[2][0].set_ylabel('$j^{off-diag}(t)$ A/cm$^2$')
        self.ax[0][j].set_title('xyz'[j])
        self.ax[-1][j].set_xlabel('t (fs)')

    def _plot_data(self, ax, data):
        # Plot data with optional smoothing
        data_used, timedata_used = self._smooth_data(data)
        time_range_mask = (self.mintime_plot < timedata_used) & (timedata_used < self.maxtime_plot)
        assert time_range_mask.any(), f"No data points in the specified time range after possible smooth. Please check the time range settings. mintime_plot:{self.mintime_plot}, maxtime_plot:{self.maxtime_plot},datanumber:{len(data_used)}"
        ax.plot(timedata_used[time_range_mask], data_used[time_range_mask])

    def _smooth_data(self, data):
        # Apply smoothing to the data if enabled
        data_used, timedata_used = data.copy(), self.timedata.copy()
        for _ in range(self.configsetting.smooth_times):
            if self.configsetting.smooth_on:
                if self.configsetting.smooth_method == 'savgol':
                    data_used = savgol_filter(data_used, 500, 3)
                elif self.configsetting.smooth_method == 'flattop':
                    window = sgl.flattop(self.configsetting.smooth_windowlen, sym=False)
                    data_used = np.convolve(data_used, window, mode='valid') / window.sum()
                    mid = len(self.timedata) // 2
                    timedata_used = self.timedata[mid - len(data_used) // 2: mid + (len(data_used) + 1) // 2]
        return data_used, timedata_used

def do(DMDana_ini_config_setting: DMDana_ini_config_setting_class):
    config = config_current(funcname='current_plot', DMDana_ini_config_setting=DMDana_ini_config_setting, show_init_log=True)
    plotter = plot_current_plot(config)
    
    if config.configsetting.plot_all:
        config.configsetting.smooth_on = False
        plotter.plot()
        config.configsetting.smooth_on = True
        plotter.plot()
    else:
        plotter.plot()
