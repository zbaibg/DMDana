#! /usr/bin/env python
"""_summary_
This script plots the current figures
"""
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal.windows as sgl
from scipy.signal import savgol_filter

from ..lib import constant as const
from .config import DMDana_ini_Class, config_current


#Read input
class param_class(object):
    def __init__(self,DMDana_ini: DMDana_ini_Class):#init_from_config, for single folder analysis
        self.DMDana_ini: DMDana_ini_Class=DMDana_ini
        self.folderlist=DMDana_ini.folderlist
        self.folder_number=len(self.folderlist)
        self.loadcurrent(0)
        self.tmax=self.config.Input.getfloat("t_max")
        if self.tmax==-1:
            self.tmax=np.max(self.jx_data[:,0])/const.fs
        self.tmin=self.config.Input.getfloat("t_min")
        self.total_time=self.tmax-self.tmin
        self.current_plot_output=self.config.Input.get('current_plot_output')
        self.light_label=self.config.light_label
        self.only_jtot=self.config.only_jtot
        self.smooth_on=self.config.Input.getboolean('smooth_on')
        self.smooth_method=self.config.Input.get('smooth_method')
        self.smooth_windowlen=self.config.Input.getint('smooth_windowlen')
        self.plot_all=self.config.Input.getboolean('plot_all')
        self.smooth_times=self.config.Input.getint('smooth_times')
    def loadcurrent(self,filenumber):
        assert filenumber<self.folder_number and filenumber>=0
        self.config: config_current=self.DMDana_ini.get_folder_config('current_plot',filenumber)
        self.jx_data=self.config.jx_data
        self.jy_data=self.config.jy_data
        self.jz_data=self.config.jz_data

def do(DMDana_ini: DMDana_ini_Class):#multiple folder analysis
    param=param_class(DMDana_ini)
    if param.plot_all:
        param.smooth_on=False
        plot_current(param).plot()
        param.smooth_on=True
        plot_current(param).plot()
    else:
        plot_current(param).plot()
        
class plot_current:
    def __init__(self,param: param_class):
        self.param=param
        self.fig1=None
        self.timedata=None
        self.datamax=0
        self.mintime_plot=self.param.tmin-0.05*self.param.total_time
        self.maxtime_plot=self.param.tmax+0.05*self.param.total_time
        self.ax: Union[List[plt.Axes],List[List[plt.Axes]]]
    def plot(self):
        if self.param.only_jtot:
            self.fig1, self.ax = plt.subplots(1,3, figsize=(10,6),dpi=200,sharex=True)
        else:
            self.fig1, self.ax = plt.subplots(3,3, figsize=(10,6),dpi=200,sharex=True)
        for folder_i in range(self.param.folder_number):
            self.param.loadcurrent(folder_i)
            self.jx_data=self.param.jx_data
            self.jy_data=self.param.jy_data
            self.jz_data=self.param.jz_data
            self.timedata=self.jx_data[:,0]/const.fs
            if self.param.only_jtot:
                self.plot_tot()
            else:
                self.plot_tot_diag_offdiag()
        self.fig1.tight_layout()
        self.savefig()
        plt.close(self.fig1)
    def plot_tot_diag_offdiag(self):
        self.fig1.suptitle('Current'+self.param.light_label)
        for jtemp,jdirection,j in [(self.jx_data,'x',0),(self.jy_data,'y',1),(self.jz_data,'z',2)]:
            j:int
            for i in range(3):
                i:int
                self.ax[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                self.ax[i][j].yaxis.major.formatter._useMathText = True 
                self.ax[i][j].set_xlim(self.mintime_plot,self.maxtime_plot)
            self.ax[0][0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
            self.ax[1][0].set_ylabel('$j^{diag}(t)$ A/cm$^2$')
            self.ax[2][0].set_ylabel('$j^{off-diag}(t)$ A/cm$^2$')
            self.ax[0][j].set_title(jdirection)
            self.ax[-1][j].set_xlabel('t (fs)')  
            self.plot_func(self.ax[0][j],jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
            self.plot_func(self.ax[1][j],jtemp[:,2])#, label=r'$j'+jdirection+'^{diag}(t)$    polarization = '+light_label)
            self.plot_func(self.ax[2][j],jtemp[:,3])#, label=r'$j'+jdirection+'^{off-diag}(t)$    polarization = '+light_label)
        #for i in range(3):
        #    for j in range(3):
        #        self.ax[i][j].set_ylim(-self.datamax,self.datamax)
    def savefig(self):
        if not self.param.smooth_on:
            smooth_str='off'
        else:
            smooth_str='on_%s_smoothtimes_%d'%(self.param.smooth_method,self.param.smooth_times)
            if self.param.smooth_method=='flattop':
                smooth_str+='_windowlen_%d'%self.param.smooth_windowlen
        self.fig1.savefig("j_smooth_%s.png"%smooth_str)
    def plot_tot(self):
        self.fig1.suptitle('Current'+self.param.light_label)
        for jtemp,jdirection,j in [(self.jx_data,'x',0),(self.jy_data,'y',1),(self.jz_data,'z',2)]:
            j:int
            self.ax[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            self.ax[j].yaxis.major.formatter._useMathText = True 
            self.ax[0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
            self.ax[j].set_title(jdirection)
            self.ax[j].set_xlabel('t (fs)')  
            self.plot_func(self.ax[j],jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
            self.ax[j].set_xlim(self.mintime_plot,self.maxtime_plot)
        #for i in range(3):
        #    self.ax[j].set_ylim(-self.datamax,self.datamax)
    def plot_func(self,ax: plt.Axes,data):
        windowlen=self.param.smooth_windowlen
        windowdata=sgl.flattop(windowlen, sym=False)
        data_used=data
        timedata_used=self.timedata
        for i in range(self.param.smooth_times):
            if self.param.smooth_on:
                if self.param.smooth_method=='savgol':
                    data_used=savgol_filter(data_used, 500, 3)
                    timedata_used=self.timedata
                elif self.param.smooth_method=='flattop': 
                    data_used=np.convolve(data_used,windowdata,mode='valid')/np.sum(windowdata)     
                    timedata_used=self.timedata[len(self.timedata)//2-len(data_used)//2:len(self.timedata)//2+(len(data_used)+1)//2]
        #self.datamax=np.max([np.max(abs(data)),self.datamax])
        time_range_mask=np.logical_and(timedata_used>self.mintime_plot,timedata_used<self.maxtime_plot)
        ax.plot(timedata_used[time_range_mask], data_used[time_range_mask])