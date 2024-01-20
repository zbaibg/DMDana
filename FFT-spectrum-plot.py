#! /usr/bin/env python
"""_summary_
This script plots FFT spectra with different parameters in batch.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal.windows as sgl
import itertools
import pandas as pd
import configparser
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('DMDana.ini')
Input=config['FFT-spectrum-plot']
fs  = 41.341373335
Hatree_to_eV = 27.211386245988

only_jtot=Input.getboolean('only_jtot')
if only_jtot==None:
    raise ValueError('only_jtot is not correct setted.')

Cutoff_list= [int(i) for i in Input['Cutoff_list'].split(',')]#it counts the number of pieces in jx(yz)_elec_tot.out
Window_type_list=[i.strip() for i in Input['Window_type_list'].split(',')]  # Rectangular, Flattop, Hann, Hamming 
light_label=' '+Input['light_label']
jx_data = np.loadtxt(Input['jx_data'],skiprows=1)
jy_data = np.loadtxt(Input['jy_data'],skiprows=1)
jz_data = np.loadtxt(Input['jz_data'],skiprows=1)
if jx_data.shape[0]!= jy_data.shape[0] or jx_data.shape[0]!= jz_data.shape[0] or jy_data.shape[0]!= jz_data.shape[0]:
    raise ValueError('The line number in jx_data jy_data jz_data are not the same. Please deal with your data.' )
Log_y_scale=Input.getboolean('Log_y_scale')
if Log_y_scale==None:
    Log_y_scale=True

Summary_output_csv=Input.getboolean('Summary_output_csv')
Summary_output_xlsx=Input.getboolean('Summary_output_xlsx')
Summary_output_filename_csv=Input['Summary_output_filename_csv']
Summary_output_filename_xlsx=Input['Summary_output_filename_xlsx']

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


if not only_jtot:
    #Plot FFT spectra
    for Window_type,Cutoff in list(itertools.product(Window_type_list,Cutoff_list)):
        paramdict=dict(Cutoff=Cutoff,Window_type=Window_type)
        output_prefix=''
        for name in paramdict:
            output_prefix=output_prefix+name+'='
            if type(paramdict[name])==int:
                output_prefix=output_prefix+'%.3e'%(paramdict[name])+';'
            else:
                output_prefix=output_prefix+str(paramdict[name])+';'

        fig2, ax2 = plt.subplots(3,3,figsize=(10,6),dpi=200, sharex=True)
        fig2.suptitle('FFT Spectrum of Current'+light_label)
        for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
            # FFT of different contributions
            # as it is clear from the above graphs that there a transient at 
            # at the start of dynamics. The transient can be removed by choosing 
            # appropriate cutoff below
            f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff)
            f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff)
            f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff)
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
else:
    #Plot FFT spectra
    for Window_type,Cutoff in list(itertools.product(Window_type_list,Cutoff_list)):
        paramdict=dict(Cutoff=Cutoff,Window_type=Window_type)
        output_prefix=''
        for name in paramdict:
            output_prefix=output_prefix+name+'='
            if type(paramdict[name])==int:
                output_prefix=output_prefix+'%.3e'%(paramdict[name])+';'
            else:
                output_prefix=output_prefix+str(paramdict[name])+';'
        fig2, ax2 = plt.subplots(1,3,figsize=(10,6),dpi=200, sharex=True)
        fig2.suptitle('FFT Spectrum of Current'+light_label)
        for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
            # FFT of different contributions
            # as it is clear from the above graphs that there a transient at 
            # at the start of dynamics. The transient can be removed by choosing 
            # appropriate cutoff below
            f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff)
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
        f_tot, jw_tot = fft_of_j(jtemp[:,0:2:1], Cutoff)
        if not only_jtot:
            f_d, jw_d = fft_of_j(jtemp[:,0:3:2], Cutoff)
            f_od, jw_od = fft_of_j(jtemp[:,0:4:3], Cutoff)
        if only_jtot:
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
database=database.transpose()
if Summary_output_csv:
    database.to_csv(Summary_output_filename_csv)
if Summary_output_xlsx:
    database.to_excel(Summary_output_filename_xlsx)

