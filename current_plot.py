#! /usr/bin/env python
"""_summary_
This script plots the current figures
"""

import matplotlib.pyplot as plt
from constant import *
import numpy as np
from global_variable import config
#Read input
tmax=config.Input.getint("t_max",-1)
jx_data=config.jx_data
jy_data=config.jy_data
jz_data=config.jz_data
if tmax==-1:
    tmax=np.max(jx_data[:,0])/fs
tmin=config.Input.getint("t_min",0)
total_time=tmax-tmin 
current_plot_output=config.Input.get('current_plot_output',"j.png")
light_label=config.light_label
only_jtot=config.only_jtot

def do():
    if only_jtot:
        plot_tot()
    else:
        plot_tot_diag_offdiag()
        
def plot_tot_diag_offdiag():
    fig1, ax1 = plt.subplots(3,3, figsize=(10,6),dpi=200,sharex=True)
    fig1.suptitle('Current'+light_label)
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
        ax1[0][j].plot(jtemp[:,0]/fs, jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
        ax1[1][j].plot(jtemp[:,0]/fs, jtemp[:,2])#, label=r'$j'+jdirection+'^{diag}(t)$    polarization = '+light_label)
        ax1[2][j].plot(jtemp[:,0]/fs, jtemp[:,3])#, label=r'$j'+jdirection+'^{off-diag}(t)$    polarization = '+light_label)
    fig1.tight_layout()
    fig1.savefig(current_plot_output)
    plt.close(fig1)
    
def plot_tot():
    fig1, ax1 = plt.subplots(1,3, figsize=(10,6),dpi=200,sharex=True)
    fig1.suptitle('Current'+light_label)
    for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
        ax1[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1[j].yaxis.major.formatter._useMathText = True 
        ax1[0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
        ax1[j].set_title(jdirection)
        ax1[j].set_xlabel('t (fs)')  
        ax1[j].plot(jtemp[:,0]/fs, jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
        ax1[j].set_xlim(tmin-0.05*total_time,tmax+0.05*total_time)
    fig1.tight_layout()
    fig1.savefig(current_plot_output)
    plt.close(fig1)