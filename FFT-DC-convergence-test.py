#! /usr/bin/env python
"""_summary_
This script plots the DC components of FFT results with different parameters in batch. It also output the data to text.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sgl
import itertools
import pandas as pd
from constant import *
import config
config.init('FFT-DC-convergence-test')



Window_type_list=[i.strip() for i in config.Input['Window_type_list'].split(',')]  # Rectangular, Flattop, Hann, Hamming 
Database_output_filename_csv=config.Input['Database_output_filename_csv']
Database_output_filename_xlsx=config.Input['Database_output_filename_xlsx']
Database_output_csv=config.Input.getboolean('Database_output_csv')
Database_output_xlsx=config.Input.getboolean('Database_output_xlsx')
Figure_output_filename=config.Input['Figure_output_filename']
Cutoff_min=config.Input.getint('Cutoff_min')#These "Cutoff" values counts the number of pieces in jx(yz)_elec_tot.out
Cutoff_max=config.Input.getint('Cutoff_max')
if Cutoff_max==-1:
    Cutoff_max=config.jx_data.shape[0]-1
Cutoff_step=config.Input.getint('Cutoff_step')
Cutoff_list= range(Cutoff_min,Cutoff_max,Cutoff_step)

# funciton which performs FFT, 
# shifts frequency bins to only plot positive frequencies, 
# changes bins to physical units (eV), applies window to time domain data, 
# and returns a normalized FFT
def fft_of_j(j_t, cutoff):
    dt = j_t[1,0] - j_t[0,0]
    N_jt = j_t[cutoff:,1].shape[0]
    freq_bins = np.fft.fftfreq(N_jt, dt)*(2.0*np.pi*Hatree_to_eV)
    shifted_freq_bins = freq_bins[:len(freq_bins)//2]

    if Window_type=='Flattop':#Rectangular, Flattop, Hann, Hamming 
        window = sgl.flattop(N_jt, sym=False)
    elif Window_type=='Hann':
        window = sgl.hann(N_jt, sym=False)
    elif Window_type=='Hamming':
        window = sgl.hamming(N_jt,sym=False)
    elif Window_type=='Rectangular':
        window= 1
    else:
        raise ValueError('Window type is not supported')
    win_jt = window*j_t[cutoff:,1]
    
    fft = np.fft.fft(win_jt)/np.mean(window)
    shifted_fft = fft[:N_jt//2]
    return shifted_freq_bins, (1/N_jt)*(shifted_fft)

#Calculate FFT DC results and output the data
database=pd.DataFrame()
for Window_type,Cutoff in list(itertools.product(Window_type_list,Cutoff_list)):
    paramdict=dict(Cutoff=Cutoff,FFT_integral_start_time_fs=config.jx_data[Cutoff,0]/fs,FFT_integral_end_time_fs=config.jx_data[-1,0]/fs,Window_type=Window_type)
    database_newline_index=database.shape[0]
    for jtemp,jdirection,j in [(config.jx_data,'x',0),(config.jy_data,'y',1),(config.jz_data,'z',2)]:
        
        # FFT of different contributions
        # as it is clear from the above graphs that there a transient at 
        # at the start of dynamics. The transient can be removed by choosing 
        # appropriate cutoff below
        f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff)
        if not config.only_jtot:
            f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff)
            f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff)
        if config.only_jtot:
            resultdisc={'FFT(j'+jdirection+'_tot)(0)':abs(jw_tot[0]),
                    'j'+jdirection+'_tot_mean': np.mean(jtemp[Cutoff:,1]),
                    'time(fs)':jtemp[Cutoff,0]/fs}
        else:
            resultdisc={'FFT(j'+jdirection+'_tot)(0)':abs(jw_tot[0]),
                    'FFT(j'+jdirection+'_d)(0)': abs(jw_d[0]),
                    'FFT(j'+jdirection+'_od)(0)': abs(jw_od[0]),
                    'j'+jdirection+'_tot_mean': np.mean(jtemp[Cutoff:,1])}
        
        database.loc[database_newline_index,list(paramdict)]=list(paramdict.values())
        database.loc[database_newline_index,list(resultdisc)]=list(resultdisc.values())
if Database_output_csv:
    database.to_csv(Database_output_filename_csv)
if Database_output_xlsx:
    database.to_excel(Database_output_filename_xlsx)

#Plot the FFT DC results
if config.only_jtot:
    fig3,ax3=plt.subplots(1,3,figsize=(16,9),dpi=200)
    for win_type in Window_type_list:
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jx_tot)(0)',ax=ax3[0],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jy_tot)(0)',ax=ax3[1],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jz_tot)(0)',ax=ax3[2],logy=True,label=win_type,xlabel='cutoff time/fs')
    ax3[0].set_title('x')
    ax3[1].set_title('y')
    ax3[2].set_title('z')
    ax3[0].set_ylabel('FFT($j_{tot}$)(0) A/cm$^2$')
    fig3.tight_layout()
    fig3.savefig(Figure_output_filename)
    plt.close(fig3)

else:
    fig3,ax3=plt.subplots(3,3,figsize=(16,9),dpi=200)
    for win_type in Window_type_list:
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jx_tot)(0)',ax=ax3[0,0],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jy_tot)(0)',ax=ax3[0,1],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jz_tot)(0)',ax=ax3[0,2],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jx_d)(0)',ax=ax3[1,0],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jy_d)(0)',ax=ax3[1,1],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jz_d)(0)',ax=ax3[1,2],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jx_od)(0)',ax=ax3[2,0],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jy_od)(0)',ax=ax3[2,1],logy=True,label=win_type,xlabel='cutoff time/fs')
        database[(database.Window_type==win_type)].plot('FFT_integral_start_time_fs','FFT(jz_od)(0)',ax=ax3[2,2],logy=True,label=win_type,xlabel='cutoff time/fs')
    ax3[0,0].set_title('x')
    ax3[0,1].set_title('y')
    ax3[0,2].set_title('z')
    ax3[0,0].set_ylabel('FFT($j_{tot}$)(0) A/cm$^2$')
    ax3[1,0].set_ylabel('FFT($j_{d}$)(0) A/cm$^2$')
    ax3[2,0].set_ylabel('FFT($j_{od}$)(0) A/cm$^2$')

    for i in range(2):
        for j in range(3):
            ax3[i,j].set_xlabel('')
    fig3.tight_layout()
    fig3.savefig(Figure_output_filename)
    plt.close(fig3)


config.end()