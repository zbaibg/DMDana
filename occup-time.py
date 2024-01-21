#! /usr/bin/env python
"""_summary_
This script plot the occupation functions with time, f(E,t). It also plots the E-axis maximum for the time derivative of f(E,t), (df/dt)max_in_E_axis.
This program does not read time from file content for now. It only count the file numbers to get time. 
So be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
import matplotlib.pyplot as plt
from init import init
init('occup-time')
from init import *
#This much be done after running initialization function in order to import variables correctly
fig = plt.figure(figsize=(12, 10))
ax =fig.add_subplot(111,projection='3d')
for index, file in enumerate(occup_selected_files):
    if index*occup_timestep_for_selected_file_ps>occup_t_tot/1000:
        break
    data = np.loadtxt(file)
    ax.scatter(data[:,0]*Hatree_to_eV, data[:,1], zs=index*occup_timestep_for_selected_file_ps, zdir='y')
ax.set_xlim(0.05*Hatree_to_eV, 0.06*Hatree_to_eV)
ax.set_ylim(0, occup_t_tot/1000)
ax.set_zlim(0, 1)

ax.set_xlabel('E (eV)')
ax.set_ylabel('t (ps)')
ax.set_zlabel('f')
ax.xaxis.set_pane_color('w')
ax.yaxis.set_pane_color('w')
ax.zaxis.set_pane_color('w')

ax.view_init(elev=20., azim=-70, roll=0)
fig.savefig('occup_time_Ttot%dfs_Step%dfs.png'%(occup_t_tot,occup_timestep_for_selected_file_ps*1000), bbox_inches="tight")

end()