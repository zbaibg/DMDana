#! /usr/bin/env python
"""_summary_
This script plots the current figures
"""

import matplotlib.pyplot as plt
from constant import *
import numpy as np
from global_variable import config
tmax=config.Input.get("tmax",-1)
if tmax==-1:
    tmax=np.max(config.jx_data[:,0])/fs
tmin=config.Input.get("tmin",0)
current_plot_output=config.Input.get('current_plot_output',"j.png")
def do():
    #Plot Current
    if not config.only_jtot:
        fig1, ax1 = plt.subplots(3,3, figsize=(10,6),dpi=200,sharex=True)
        fig1.suptitle('Current'+config.light_label)
        for jtemp,jdirection,j in [(config.jx_data,'x',0),(config.jy_data,'y',1),(config.jz_data,'z',2)]:
            for i in range(3):
                ax1[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                ax1[i][j].yaxis.major.formatter._useMathText = True 
                total_time=tmax-tmin 
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
    else:
        fig1, ax1 = plt.subplots(1,3, figsize=(10,6),dpi=200,sharex=True)
        fig1.suptitle('Current'+config.light_label)
        for jtemp,jdirection,j in [(config.jx_data,'x',0),(config.jy_data,'y',1),(config.jz_data,'z',2)]:
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