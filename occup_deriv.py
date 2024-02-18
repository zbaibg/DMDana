#! /usr/bin/env python
"""_summary_
This script plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.
(Using second center finite difference).
This program does not read time from file content for now. It only count the file numbers to get time. 
So be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
import matplotlib.pyplot as plt
from constant import *
from global_variable import config
def do():
    n = len(config.occup_selected_files)#number of files left in "occup_selected_files"
    data= np.full((n, 5646), np.nan)# why 5646?
    dfdt = np.full((n,5646), np.nan)
    dfdtMax = np.full((n), np.nan)
    tarray=np.array(range(n))*config.occup_timestep_for_selected_file_fs/1000#ps
    #note that t in occupations_t0.out is actually not 0
    #but if occup_timestep_for_selected_file_fs is much larger than t0,
    #it is fine to take t0 as 0 as it does here. 
    for ind in range(n):
        data[ind,:] = np.loadtxt(config.occup_selected_files[ind])[:,1]
    dfdt=np.gradient(data,config.occup_timestep_for_selected_file_fs,axis=0)#df/fs
    dfdtMax = np.max(dfdt,axis=1)
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.plot(tarray[0:config.occup_maxmium_file_number_plotted_exclude_t0+1], dfdtMax[0:config.occup_maxmium_file_number_plotted_exclude_t0+1], '--.')
    ax4.set_yscale('log')
    ax4.set_xlabel('t (ps)')
    ax4.set_ylabel(r'max$[\frac{df}{dt}]$ unit:df/fs')
    fig4.savefig('dfdt_max_Ttot%dfs_Step%dfs.png'%(config.occup_t_tot,config.occup_timestep_for_selected_file_fs))