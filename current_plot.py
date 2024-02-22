#! /usr/bin/env python
"""_summary_
This script plots the current figures
"""
import scipy.signal.windows as sgl
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from constant import *
import numpy as np
from global_variable import config
#Read input
tmax=config.Input.getint("t_max")
jx_data=config.jx_data
jy_data=config.jy_data
jz_data=config.jz_data
if tmax==-1:
    tmax=np.max(jx_data[:,0])/fs
tmin=config.Input.getint("t_min")
total_time=tmax-tmin 
current_plot_output=config.Input.get('current_plot_output')
light_label=config.light_label
only_jtot=config.only_jtot
smooth_on=config.Input.getboolean('smooth_on')
smooth_method=config.Input.get('smooth_method')
smooth_windowlen=config.Input.getint('smooth_windowlen')
def do():
    temp=plot_current()
    temp.plot()
        
class plot_current:
    def __init__(self):
        self.fig1=None
        self.timedata=jx_data[:,0]/fs
        self.datamax=0
    def plot(self):
        if only_jtot:
            self.plot_tot()
        else:
            self.plot_tot_diag_offdiag()
    def plot_tot_diag_offdiag(self):
        self.fig1, ax1 = plt.subplots(3,3, figsize=(10,6),dpi=200,sharex=True)
        self.fig1.suptitle('Current'+light_label)
        for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
            for i in range(3):
                ax1[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                ax1[i][j].yaxis.major.formatter._useMathText = True 
                ax1[i][j].set_xlim(tmin-0.05*total_time,tmax+0.05*total_time)
                
            ax1[0][0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
            ax1[1][0].set_ylabel('$j^{diag}(t)$ A/cm$^2$')
            ax1[2][0].set_ylabel('$j^{off-diag}(t)$ A/cm$^2$')
            ax1[0][j].set_title(jdirection)
            ax1[-1][j].set_xlabel('t (fs)')  
            self.plot_func(ax1[0][j],jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
            self.plot_func(ax1[1][j],jtemp[:,2])#, label=r'$j'+jdirection+'^{diag}(t)$    polarization = '+light_label)
            self.plot_func(ax1[2][j],jtemp[:,3])#, label=r'$j'+jdirection+'^{off-diag}(t)$    polarization = '+light_label)
        #for i in range(3):
        #    for j in range(3):
        #        ax1[i][j].set_ylim(-self.datamax,self.datamax)
        self.fig1.tight_layout()
        self.fig1.savefig(current_plot_output)
        plt.close(self.fig1)
        
    def plot_tot(self):
        self.fig1, ax1 = plt.subplots(1,3, figsize=(10,6),dpi=200,sharex=True)
        self.fig1.suptitle('Current'+light_label)
        for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
            ax1[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax1[j].yaxis.major.formatter._useMathText = True 
            ax1[0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
            ax1[j].set_title(jdirection)
            ax1[j].set_xlabel('t (fs)')  
            self.plot_func(ax1[j],jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
            ax1[j].set_xlim(tmin-0.05*total_time,tmax+0.05*total_time)
            
        #for i in range(3):
        #    ax1[j].set_ylim(-self.datamax,self.datamax)
        self.fig1.tight_layout()
        self.fig1.savefig(current_plot_output)
        plt.close(self.fig1)
        
    def plot_func(self,ax,data):
        windowlen=smooth_windowlen
        windowdata=sgl.flattop(windowlen, sym=False)
        if smooth_on:
            if smooth_method=='savgol':
                data=savgol_filter(data, 500, 3)
            elif smooth_method=='window': 
                size=(max(len(data), len(windowdata)) - min(len(data), len(windowdata)) + 1)
                data=np.convolve(data,windowdata,mode='valid')/np.sum(windowdata)   
                self.timedata_cut=self.timedata[int(windowlen/2):len(self.timedata)-int(windowlen/2)+1]
        #self.datamax=np.max([np.max(abs(data)),self.datamax])
        ax.plot(self.timedata_cut, data)