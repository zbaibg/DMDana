#! /usr/bin/env python
"""_summary_
This script plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.
(Using second center finite difference).
This program does not read time from file content for now. It only count the file numbers to get time. 
So be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
from ..lib import constant as const
from .config import config_occup,DMDana_ini_Class
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#Read input
class param_class(object):
    def __init__(self,config: config_occup):
        self.folder=config.folder
        self.occup_selected_files=config.occup_selected_files
        self.occup_timestep_for_selected_file_fs=config.occup_timestep_for_selected_file_fs
        self.occup_maxmium_file_number_plotted_exclude_t0=config.occup_maxmium_file_number_plotted_exclude_t0
        self.occup_t_tot=config.occup_t_tot
        self.data_first=np.loadtxt(self.folder+'/occupations_t0.out')
def do(DMDana_ini:DMDana_ini_Class):
    config=DMDana_ini.get_folder_config('occup_deriv',0)
    param=param_class(config)
    plot_object=occup_deriv(param)
    plot_object.do()
class occup_deriv(object):
    def __init__(self,param: param_class):
        self.param=param
    def do(self):
        n = len(self.param.occup_selected_files)#number of files left in "occup_selected_files"
        data= np.full((n, len(self.param.data_first)), np.nan)# why 5646?
        dfdt = np.full((n,len(self.param.data_first)), np.nan)
        dfdtMax = np.full((n), np.nan)
        tarray=np.array(range(n))*self.param.occup_timestep_for_selected_file_fs/1000#ps
        #note that t in occupations_t0.out is actually not 0
        #but if occup_timestep_for_selected_file_fs is much larger than t0,
        #it is fine to take t0 as 0 as it does here. 
        for ind in range(n):
            data[ind,:] = np.loadtxt(self.param.occup_selected_files[ind])[:,1]
        dfdt=np.gradient(data,self.param.occup_timestep_for_selected_file_fs,axis=0)#df/fs
        dfdtMax = np.max(dfdt,axis=1)
        ax: plt.Axes
        fig: Figure
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(tarray[0:self.param.occup_maxmium_file_number_plotted_exclude_t0+1], dfdtMax[0:self.param.occup_maxmium_file_number_plotted_exclude_t0+1], '--.')
        ax.set_yscale('log')
        ax.set_xlabel('t (ps)')
        ax.set_ylabel(r'max$[\frac{df}{dt}]$ unit:df/fs')
        fig.savefig('dfdt_max_Ttot%dfs_Step%dfs.png'%(self.param.occup_t_tot,self.param.occup_timestep_for_selected_file_fs))
        plt.close()