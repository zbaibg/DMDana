#! /usr/bin/env python
"""_summary_
This script plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.
(Using second center finite difference).
This program does not read time from file content for now. It only count the file numbers to get time. 
So be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from pydantic.dataclasses import dataclass

from ..lib import constant as const
from .config import DMDana_ini_config_setting_class, config_occup


#Read input
@dataclass
class config_occup_deriv(config_occup):
    data_first: object = None # numpy array
    def __post_init__(self):
        self.funcname='occup_deriv'
        super().__post_init__()
        self.data_first=np.loadtxt(self.folder+'/occupations_t0.out')
def do(DMDana_ini_config_setting:DMDana_ini_config_setting_class):
    config=config_occup_deriv(DMDana_ini_config_setting=DMDana_ini_config_setting)
    plot_object=plot_occup_deriv(config)
    plot_object.do()
class plot_occup_deriv(object):
    def __init__(self,param: config_occup_deriv):
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
            try:
                data[ind,:] = np.loadtxt(self.param.occup_selected_files[ind])[:,1]
            except Exception as e:
                logging.error('cannot read file: %s'%(self.param.occup_selected_files[ind]))

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