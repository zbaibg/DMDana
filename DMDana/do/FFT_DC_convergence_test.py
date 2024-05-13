#! /usr/bin/env python
"""
This plots the change of the Direct Current component calculated by different FFT-time-range and FFT-window-functions. This aims to check FFT convergence. It could also output the analysis results to files.
"""
import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass

from ..lib import constant as const
from ..lib.fft import fft_of_j
from .config import DMDana_ini_config_setting_class, config_current


#Read input
@dataclass
class config_FFT_DC_convergence_test(config_current):
    Cutoff_max: int = None
    Cutoff_list: List[int] = None

    def __post_init__(self):
        self.funcname='FFT_DC_convergence_test'
        super().__post_init__()
        if self.Cutoff_max is None and self.configsetting.Cutoff_max <= 0:
            self.Cutoff_max = self.jx_data.shape[0] - 1
        self.Cutoff_list = range(self.configsetting.Cutoff_min, self.Cutoff_max, self.configsetting.Cutoff_step)
        
def do(config:config_FFT_DC_convergence_test):
    plot_FFT_DC_convergence_test(config).do()
class plot_FFT_DC_convergence_test(object):
    def __init__(self,param: config_FFT_DC_convergence_test):
        self.param=param
    def do(self):
        database=self.calculate_and_output_database()
        self.plot(database)
        
    def calculate_and_output_database(self):
        #Calculate FFT DC results and output the data
        database=pd.DataFrame(dtype=object)
        for Window_type,Cutoff in list(itertools.product(self.param.configsetting.Window_type_list,self.param.Cutoff_list)):
            paramdict=dict(Cutoff=Cutoff,FFT_integral_start_time_fs=self.param.jx_data[Cutoff,0]/const.fs,FFT_integral_end_time_fs=self.param.jx_data[-1,0]/const.fs,Window_type=Window_type)
            database_newline_index=database.shape[0]
            for jtemp,jdirection,j in [(self.param.jx_data,'x',0),(self.param.jy_data,'y',1),(self.param.jz_data,'z',2)]:
                
                # FFT of different contributions
                # as it is clear from the above graphs that there a transient at 
                # at the start of dynamics. The transient can be removed by choosing 
                # appropriate cutoff below
                f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff,Window_type)
                if not self.param.configsetting.only_jtot:
                    f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff,Window_type)
                    f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff,Window_type)
                if self.param.configsetting.only_jtot:
                    resultdisc={'FFT(j'+jdirection+'_tot)(0)':np.real(jw_tot[0]),
                            'j'+jdirection+'_tot_mean': np.mean(jtemp[Cutoff:,1]),
                            'time(fs)':jtemp[Cutoff,0]/const.fs}
                else:
                    resultdisc={'FFT(j'+jdirection+'_tot)(0)':np.real(jw_tot[0]),
                            'FFT(j'+jdirection+'_d)(0)': np.real(jw_d[0]),
                            'FFT(j'+jdirection+'_od)(0)': np.real(jw_od[0]),
                            'j'+jdirection+'_tot_mean': np.mean(jtemp[Cutoff:,1])}
                
                database.loc[database_newline_index,list(paramdict)]=list(paramdict.values())
                database.loc[database_newline_index,list(resultdisc)]=list(resultdisc.values())
        if self.param.configsetting.Database_output_csv:
            database.to_csv(self.param.configsetting.Database_output_filename_csv)
        if self.param.configsetting.Database_output_xlsx:
            database.to_excel(self.param.configsetting.Database_output_filename_xlsx)
        return database
    def plot(self,database):
        #Plot the FFT DC results
        if self.param.configsetting.only_jtot:
            self.plot_tot(database)
        else:
            self.plot_tot_diag_offdiag(database)
    def plot_tot(self,database):
        fig3,ax=plt.subplots(1,3,figsize=(16,9),dpi=200)
        ax: List[plt.Axes]
        for win_type in self.param.configsetting.Window_type_list:
            plottime=database[(database.Window_type==win_type)]['FFT_integral_start_time_fs']
            plotjx_tot=database[(database.Window_type==win_type)]['FFT(jx_tot)(0)']
            plotjy_tot=database[(database.Window_type==win_type)]['FFT(jy_tot)(0)']
            plotjz_tot=database[(database.Window_type==win_type)]['FFT(jz_tot)(0)']
            ax[0].plot(plottime,np.abs(plotjx_tot),label=win_type)
            ax[1].plot(plottime,np.abs(plotjy_tot),label=win_type)
            ax[2].plot(plottime,np.abs(plotjz_tot),label=win_type)
        ax[0].set_title('x')
        ax[1].set_title('y')
        ax[2].set_title('z')
        ax[0].set_ylabel('abs[FFT($j_{tot}$)(0)] A/cm$^2$')
        for i in range(3):
            ax[i].set_yscale('log')
            ax[i].set_xlabel('cutoff time/fs')
            ax[i].legend()
        fig3.tight_layout()
        fig3.savefig(self.param.configsetting.Figure_output_filename)
        plt.close(fig3)
    def plot_tot_diag_offdiag(self,database):
            fig,ax=plt.subplots(3,3,figsize=(16,9),dpi=200)
            ax: List[List[plt.Axes]]
            for win_type in self.param.configsetting.Window_type_list:
                plottime=database[(database.Window_type==win_type)]['FFT_integral_start_time_fs']
                plotjx_tot=database[(database.Window_type==win_type)]['FFT(jx_tot)(0)']
                plotjy_tot=database[(database.Window_type==win_type)]['FFT(jy_tot)(0)']
                plotjz_tot=database[(database.Window_type==win_type)]['FFT(jz_tot)(0)']
                plotjx_d=database[(database.Window_type==win_type)]['FFT(jx_d)(0)']
                plotjy_d=database[(database.Window_type==win_type)]['FFT(jy_d)(0)']
                plotjz_d=database[(database.Window_type==win_type)]['FFT(jz_d)(0)']       
                plotjx_od=database[(database.Window_type==win_type)]['FFT(jx_od)(0)']
                plotjy_od=database[(database.Window_type==win_type)]['FFT(jy_od)(0)']
                plotjz_od=database[(database.Window_type==win_type)]['FFT(jz_od)(0)']    
                ax[0][0].plot(plottime,np.abs(plotjx_tot),label=win_type)
                ax[0][1].plot(plottime,np.abs(plotjy_tot),label=win_type)
                ax[0][2].plot(plottime,np.abs(plotjz_tot),label=win_type)
                ax[1][0].plot(plottime,np.abs(plotjx_d),label=win_type)
                ax[1][1].plot(plottime,np.abs(plotjy_d),label=win_type)
                ax[1][2].plot(plottime,np.abs(plotjz_d),label=win_type)
                ax[2][0].plot(plottime,np.abs(plotjx_od),label=win_type)
                ax[2][1].plot(plottime,np.abs(plotjy_od),label=win_type)
                ax[2][2].plot(plottime,np.abs(plotjz_od),label=win_type)
            ax[0][0].set_title('x')
            ax[0][1].set_title('y')
            ax[0][2].set_title('z')
            ax[0][0].set_ylabel('abs[FFT($j_{tot}$)(0)] A/cm$^2$')
            ax[1][0].set_ylabel('abs[FFT($j_{d}$)(0)] A/cm$^2$')
            ax[2][0].set_ylabel('abs[FFT($j_{od}$)(0)] A/cm$^2$')
            for i in range(3):
                ax[2][i].set_xlabel('cutoff time/fs')
                for j in range(3):
                    ax[i][j].set_yscale('log')
                    ax[i][j].legend()
            fig.tight_layout()
            fig.savefig(self.param.configsetting.Figure_output_filename)
            plt.close(fig)
            
