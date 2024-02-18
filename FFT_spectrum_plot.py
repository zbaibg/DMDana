#! /usr/bin/env python
"""_summary_
This script plots FFT spectra with different parameters in batch.
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from constant import *
from global_variable import config
from common import fft_of_j
#Read input
Cutoff_list= [int(i) for i in config.Input['Cutoff_list'].split(',')]#it counts the number of pieces in jx(yz)_elec_tot.out
Window_type_list=[i.strip() for i in config.Input['Window_type_list'].split(',')]  # Rectangular, Flattop, Hann, Hamming 
Log_y_scale=config.Input.getboolean('Log_y_scale')
if Log_y_scale==None:
    Log_y_scale=True
Summary_output_csv=config.Input.getboolean('Summary_output_csv')
Summary_output_xlsx=config.Input.getboolean('Summary_output_xlsx')
Summary_output_filename_csv=config.Input['Summary_output_filename_csv']
Summary_output_filename_xlsx=config.Input['Summary_output_filename_xlsx']
only_jtot=config.only_jtot
light_label=config.light_label
jx_data=config.jx_data
jy_data=config.jy_data
jz_data=config.jz_data


def do():
    plot()
    output_database()
    
def plot():
    #Plot FFT spectra
    for Window_type,Cutoff in list(itertools.product(Window_type_list,Cutoff_list)):
        paramdict=dict(Cutoff=Cutoff,Window_type=Window_type)
        output_prefix=''
        for name in paramdict:
            output_prefix=output_prefix+name+'='
            if type(paramdict[name])==int:
                output_prefix=output_prefix+'%d'%(paramdict[name])+';'
            else:
                output_prefix=output_prefix+str(paramdict[name])+';'
        if not only_jtot:
            plot_tot_diag_offdiag(Cutoff,Window_type,output_prefix)
        else:
            plot_tot(Cutoff,Window_type,output_prefix)
            
def plot_tot_diag_offdiag(Cutoff,Window_type,output_prefix):
    fig2, ax2 = plt.subplots(3,3,figsize=(10,6),dpi=200, sharex=True)
    fig2.suptitle('FFT Spectrum of Current'+light_label)
    for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
        # FFT of different contributions
        # as it is clear from the above graphs that there a transient at 
        # at the start of dynamics. The transient can be removed by choosing 
        # appropriate cutoff below
        f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff,Window_type)
        f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff,Window_type)
        f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff,Window_type)
        for i in range(3):
            ax2[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax2[i][j].yaxis.major.formatter._useMathText = True
            if Log_y_scale:
                ax2[i][j].set_yscale('log')
        ax2[0][0].set_ylabel('$\hat{j}^{tot}(\omega)$ A/cm$^2$')
        ax2[1][0].set_ylabel('$\hat{j}^{diag}(\omega)$ A/cm$^2$')
        ax2[2][0].set_ylabel('$\hat{j}^{off-diag}(\omega)$ A/cm$^2$')
        ax2[0][j].set_title(jdirection)
        ax2[-1][j].set_xlabel('$\omega$ (eV)')
        ax2[0][j].plot(f_tot, abs(jw_tot), label='total')
        ax2[1][j].plot(f_tot, abs(jw_d), label='diagonal')
        ax2[2][j].plot(f_tot, abs(jw_od), label='off-diagonal')
    fig2.tight_layout()
    fig2.savefig('./'+output_prefix+'-j-fft.png')
    plt.close(fig2)

def plot_tot(Cutoff,Window_type,output_prefix):
    fig2, ax2 = plt.subplots(1,3,figsize=(10,6),dpi=200, sharex=True)
    fig2.suptitle('FFT Spectrum of Current'+light_label)
    for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
        # FFT of different contributions
        # as it is clear from the above graphs that there a transient at 
        # at the start of dynamics. The transient can be removed by choosing 
        # appropriate cutoff below
        f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff,Window_type)
        ax2[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2[j].yaxis.major.formatter._useMathText = True
        ax2[0].set_ylabel('$\hat{j}^{tot}(\omega)$ A/cm$^2$')
        ax2[j].set_title(jdirection)
        ax2[j].set_xlabel('$\omega$ (eV)')
        ax2[j].plot(f_tot, abs(jw_tot), label='total')
        if Log_y_scale:
            ax2[j].set_yscale('log')
    fig2.tight_layout()
    fig2.savefig('./'+output_prefix+'-j-fft.png')
    plt.close(fig2)
    
def output_database():
    #Calculate information of current spectrums to output for convenience.
    database=pd.DataFrame()
    for Window_type,Cutoff in list(itertools.product(Window_type_list,Cutoff_list)):
        paramdict=dict(Cutoff=Cutoff,FFT_integral_start_time_fs=jx_data[Cutoff,0]/fs,FFT_integral_end_time_fs=jx_data[-1,0]/fs,Window_type=Window_type)
        database_newline_index=database.shape[0]
        for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
            
            # FFT of different contributions
            # as it is clear from the above graphs that there a transient at 
            # at the start of dynamics. The transient can be removed by choosing 
            # appropriate cutoff below
            f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff,Window_type)
            if not only_jtot:
                f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff,Window_type)
                f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff,Window_type)
            if only_jtot:
                resultdisc={'FFT(j'+jdirection+'_tot)(0)':np.real(jw_tot[0]),
                        'j'+jdirection+'_tot_mean': np.mean(jtemp[Cutoff:,1]),
                        'time(fs)':jtemp[Cutoff,0]/fs}
            else:
                resultdisc={'FFT(j'+jdirection+'_tot)(0)':np.real(jw_tot[0]),
                        'FFT(j'+jdirection+'_d)(0)': np.real(jw_d[0]),
                        'FFT(j'+jdirection+'_od)(0)': np.real(jw_od[0]),
                        'j'+jdirection+'_tot_mean': np.mean(jtemp[Cutoff:,1])}
            
            database.loc[database_newline_index,list(paramdict)]=list(paramdict.values())
            database.loc[database_newline_index,list(resultdisc)]=list(resultdisc.values())
    database=database.transpose()
    if Summary_output_csv:
        database.to_csv(Summary_output_filename_csv)
    if Summary_output_xlsx:
        database.to_excel(Summary_output_filename_xlsx)
        

