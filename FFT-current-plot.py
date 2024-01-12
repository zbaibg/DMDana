#!python
"""_summary_
This script plots the current figures and FFT spectra with different parameters in batch.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal.windows as sgl
import itertools
import pandas as pd
fs  = 41.341373335
sec = 4.1341373335E+16
Hatree_to_eV = 27.2114

only_jtot=True
Cutoff_list= [12000]
Window_type_list=['Rectangular', 'Flattop', 'Hann', 'Hamming']  # Rectangular, Flattop, Hann, Hamming 
light_label='$E_x$'
jx_data = np.loadtxt('./jx_elec_tot.out')
jy_data = np.loadtxt('./jy_elec_tot.out')
jz_data = np.loadtxt('./jz_elec_tot.out')

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

#Plot Current
if only_jtot:
    fig1, ax1 = plt.subplots(1,3, figsize=(10,6),dpi=200,sharex=True)
else:
    fig1, ax1 = plt.subplots(3,3, figsize=(10,6),dpi=200,sharex=True)
fig1.suptitle('Current for Light with '+light_label+' Polarizaion')
for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
    for i in range(3):
        ax1[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1[i][j].yaxis.major.formatter._useMathText = True 
    ax1[0][0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
    if not only_jtot:
        ax1[1][0].set_ylabel('$j^{diag}(t)$ A/cm$^2$')
        ax1[2][0].set_ylabel('$j^{off-diag}(t)$ A/cm$^2$')
    ax1[0][j].set_title(jdirection)
    ax1[-1][j].set_xlabel('t (fs)')  
    ax1[0][j].plot(jtemp[:,0]/fs, jtemp[:,1], label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
    if not only_jtot:
        ax1[1][j].plot(jtemp[:,0]/fs, jtemp[:,2], label=r'$j'+jdirection+'^{diag}(t)$    polarization = '+light_label)
        ax1[2][j].plot(jtemp[:,0]/fs, jtemp[:,3], label=r'$j'+jdirection+'^{off-diag}(t)$    polarization = '+light_label)
fig1.tight_layout()
fig1.savefig('./j.png')

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
    if only_jtot:
        fig2, ax2 = plt.subplots(1,3,figsize=(10,6),dpi=200, sharex=True)
    else:
        fig2, ax2 = plt.subplots(3,3,figsize=(10,6),dpi=200, sharex=True)
    fig2.suptitle('FFT of Current for Light with '+light_label+' Polarizaion')
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
            lines=1
        else:
            lines=3
        for i in range(lines):
            ax2[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax2[i][j].yaxis.major.formatter._useMathText = True
        ax2[0][0].set_ylabel('$\hat{j}^{tot}(\omega)$ A/cm$^2$')
        if not only_jtot:
            ax2[1][0].set_ylabel('$\hat{j}^{diag}(\omega)$ A/cm$^2$')
            ax2[2][0].set_ylabel('$\hat{j}^{off-diag}(\omega)$ A/cm$^2$')
        ax2[0][j].set_title(jdirection)
        ax2[-1][j].set_xlabel('$\omega$ (eV)')
        ax2[0][j].plot(f_tot, abs(jw_tot), label='total')
        if not only_jtot:
            ax2[1][j].plot(f_tot, abs(jw_d), label='diagonal')
            ax2[2][j].plot(f_tot, abs(jw_od), label='off-diagonal')
    fig2.tight_layout()
    fig2.savefig('./'+output_prefix+'-j-fft.png')
    plt.close(fig1)
    plt.close(fig2)




