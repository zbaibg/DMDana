#! /usr/bin/env python
"""_summary_
This script plot the occupation functions with time, f(E,t). It also plots the E-axis maximum for the time derivative of f(E,t), (df/dt)max_in_E_axis.
This program does not read time from file content for now. It only count the file numbers to get time. 
So be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
import matplotlib.pyplot as plt
from constant import *
import config
config.init('occup-time')

#This much be done after running initialization function in order to import variables correctly
fig = plt.figure(figsize=(12, 10))
ax =fig.add_subplot(111,projection='3d')
occup_time_plot_lowE=config.Input.getfloat('occup_time_plot_lowE')
occup_time_plot_highE=config.Input.getfloat('occup_time_plot_highE')
occup_time_plot_set_Erange=config.Input.getboolean('occup_time_plot_set_Erange')
for index, file in enumerate(config.occup_selected_files):
    if index*config.occup_timestep_for_selected_file_ps>config.occup_t_tot/1000:
        break
    data = np.loadtxt(file)
    if occup_time_plot_set_Erange:
        data=data[data[:,0].argsort()]
        data=data[np.logical_and(data[:,0]>occup_time_plot_lowE,data[:,0]<occup_time_plot_highE)]
    ax.scatter(data[:,0]*Hatree_to_eV, data[:,1], zs=index*config.occup_timestep_for_selected_file_ps, zdir='y')

if occup_time_plot_set_Erange:
    ax.set_xlim(occup_time_plot_lowE, occup_time_plot_highE)
ax.set_ylim(0, config.occup_t_tot/1000)
ax.set_zlim(0, 1)

ax.set_xlabel('E (eV)')
ax.set_ylabel('t (ps)')
ax.set_zlabel('f')
ax.xaxis.set_pane_color('w')
ax.yaxis.set_pane_color('w')
ax.zaxis.set_pane_color('w')

ax.view_init(elev=20., azim=-70, roll=0)
fig.savefig('occup_time_Ttot%dfs_Step%dfs.png'%(config.occup_t_tot,config.occup_timestep_for_selected_file_ps*1000), bbox_inches="tight")

config.end()