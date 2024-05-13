#! /usr/bin/env python
"""
This plots FFT spectra of the DMD currents. It could also output the DC components of the current FFT-setting to files
"""
import itertools
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from pydantic.dataclasses import dataclass

from ..lib import constant as const
from ..lib.fft import fft_of_j
from .config import DMDana_ini_config_setting_class, config_current


#Read input
@dataclass
class config_FFT_spectrum_plot(config_current):
    def __post_init__(self):
        self.funcname='FFT_spectrum_plot'
        super().__post_init__()
        
def do(config:config_FFT_spectrum_plot):
    plot_object=plot_FFT_spectrum_plot(config)
    plot_object.plot()
    plot_object.output_database()

class plot_FFT_spectrum_plot(object):
    def __init__(self,config: config_FFT_spectrum_plot):
        self.config=config
    def plot(self):
        #Plot FFT spectra
        #self.config.log('info',f'Cutofflist {self.config.configsetting.Cutoff_list}')
        #self.config.log('info',f'Window_type_list {self.config.configsetting.Window_type_list}')
        for Window_type,Cutoff in list(itertools.product(self.config.configsetting.Window_type_list,self.config.configsetting.Cutoff_list)):
            paramdict=dict(Cutoff=Cutoff,Window_type=Window_type)
            output_prefix=''
            for name in paramdict:
                output_prefix=output_prefix+name+'='
                if type(paramdict[name])==int:
                    output_prefix=output_prefix+'%d'%(paramdict[name])+';'
                else:
                    output_prefix=output_prefix+str(paramdict[name])+';'
            output_prefix+="log_y%s"%(str(self.config.configsetting.Log_y_scale))
            if not self.config.configsetting.only_jtot:
                self.plot_tot_diag_offdiag(Cutoff,Window_type,output_prefix)
            else:
                self.plot_tot(Cutoff,Window_type,output_prefix)
                
    def plot_tot_diag_offdiag(self,Cutoff,Window_type,output_prefix):
        ax: List[List[Axes]]
        fig, ax = plt.subplots(3,3,figsize=(10,6),dpi=200, sharex=True)
        fig.suptitle('FFT Spectrum of Current'+self.config.light_label)
        for jtemp,jdirection,j in [(self.config.jx_data,'x',0),(self.config.jy_data,'y',1),(self.config.jz_data,'z',2)]:
            j: int
            # FFT of different contributions
            # as it is clear from the above graphs that there a transient at 
            # at the start of dynamics. The transient can be removed by choosing 
            # appropriate cutoff below
            f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff,Window_type)
            f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff,Window_type)
            f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff,Window_type)
            for i in range(3):
                i: int
                ax[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                ax[i][j].yaxis.major.formatter._useMathText = True
                if self.config.configsetting.Log_y_scale:
                    ax[i][j].set_yscale('log')
            ax[0][0].set_ylabel('$\hat{j}^{tot}(\omega)$ A/cm$^2$')
            ax[1][0].set_ylabel('$\hat{j}^{diag}(\omega)$ A/cm$^2$')
            ax[2][0].set_ylabel('$\hat{j}^{off-diag}(\omega)$ A/cm$^2$')
            ax[0][j].set_title(jdirection)
            ax[-1][j].set_xlabel('$\omega$ (eV)')
            ax[0][j].plot(f_tot, abs(jw_tot), label='total')
            ax[1][j].plot(f_tot, abs(jw_d), label='diagonal')
            ax[2][j].plot(f_tot, abs(jw_od), label='off-diagonal')
        fig.tight_layout()
        fig.savefig('./'+output_prefix+'-j-fft.png')
        plt.close(fig)

    def plot_tot(self,Cutoff,Window_type,output_prefix):
        ax: List[Axes]
        fig, ax = plt.subplots(1,3,figsize=(10,6),dpi=200, sharex=True)
        fig.suptitle('FFT Spectrum of Current'+self.config.light_label)
        for jtemp,jdirection,j in [(self.config.jx_data,'x',0),(self.config.jy_data,'y',1),(self.config.jz_data,'z',2)]:
            j: int
            # FFT of different contributions
            # as it is clear from the above graphs that there a transient at 
            # at the start of dynamics. The transient can be removed by choosing 
            # appropriate cutoff below
            f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff,Window_type)
            ax[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax[j].yaxis.major.formatter._useMathText = True
            ax[0].set_ylabel('$\hat{j}^{tot}(\omega)$ A/cm$^2$')
            ax[j].set_title(jdirection)
            ax[j].set_xlabel('$\omega$ (eV)')
            ax[j].plot(f_tot, abs(jw_tot), label='total')
            if self.config.configsetting.Log_y_scale:
                ax[j].set_yscale('log')
        fig.tight_layout()
        fig.savefig('./'+output_prefix+'-j-fft.png')
        plt.close(fig)
        
    def output_database(self):
        #Calculate information of current spectrums to output for convenience.
        database=pd.DataFrame()
        for Window_type,Cutoff in list(itertools.product(self.config.configsetting.Window_type_list,self.config.configsetting.Cutoff_list)):
            paramdict=dict(Cutoff=Cutoff,FFT_integral_start_time_fs=self.config.jx_data[Cutoff,0]/const.fs,FFT_integral_end_time_fs=self.config.jx_data[-1,0]/const.fs,Window_type=Window_type)
            database_newline_index=database.shape[0]
            for jtemp,jdirection,j in [(self.config.jx_data,'x',0),(self.config.jy_data,'y',1),(self.config.jz_data,'z',2)]:
                
                # FFT of different contributions
                # as it is clear from the above graphs that there a transient at 
                # at the start of dynamics. The transient can be removed by choosing 
                # appropriate cutoff below
                f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff,Window_type)
                if not self.config.configsetting.only_jtot:
                    f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff,Window_type)
                    f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff,Window_type)
                if self.config.configsetting.only_jtot:
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
        database=database.transpose()
        if self.config.configsetting.Summary_output_csv:
            database.to_csv(self.config.configsetting.Summary_output_filename_csv)
        if self.config.configsetting.Summary_output_xlsx:
            database.to_excel(self.config.configsetting.Summary_output_filename_xlsx)
            

