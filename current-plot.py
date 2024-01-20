#!python
"""_summary_
This script plots the current figures
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sgl
import configparser
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('DMDana.ini')
Input=config['current-plot']
fs  = 41.341373335
#Hatree_to_eV = 27.211386245988
only_jtot=Input.getboolean('only_jtot')
if only_jtot==None:
    raise ValueError('only_jtot is not correct setted.')
jx_data = np.loadtxt(Input['jx_data'],skiprows=1)
jy_data = np.loadtxt(Input['jy_data'],skiprows=1)
jz_data = np.loadtxt(Input['jz_data'],skiprows=1)
if jx_data.shape[0]!= jy_data.shape[0] or jx_data.shape[0]!= jz_data.shape[0] or jy_data.shape[0]!= jz_data.shape[0]:
    raise ValueError('The line number in jx_data jy_data jz_data are not the same. Please deal with your data.' )
light_label=' '+Input['light_label']
current_plot_output=Input['current_plot_output']
#Plot Current
if not only_jtot:
    fig1, ax1 = plt.subplots(3,3, figsize=(10,6),dpi=200,sharex=True)
    fig1.suptitle('Current'+light_label)
    for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
        for i in range(3):
            ax1[i][j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            ax1[i][j].yaxis.major.formatter._useMathText = True 
        ax1[0][0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
        ax1[1][0].set_ylabel('$j^{diag}(t)$ A/cm$^2$')
        ax1[2][0].set_ylabel('$j^{off-diag}(t)$ A/cm$^2$')
        ax1[0][j].set_title(jdirection)
        ax1[-1][j].set_xlabel('t (fs)')  
        ax1[0][j].plot(jtemp[:,0]/fs, jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
        ax1[1][j].plot(jtemp[:,0]/fs, jtemp[:,2])#, label=r'$j'+jdirection+'^{diag}(t)$    polarization = '+light_label)
        ax1[2][j].plot(jtemp[:,0]/fs, jtemp[:,3])#, label=r'$j'+jdirection+'^{off-diag}(t)$    polarization = '+light_label)
    fig1.tight_layout()
    fig1.savefig(current_plot_output)
    plt.close(fig1)
else:
    fig1, ax1 = plt.subplots(1,3, figsize=(10,6),dpi=200,sharex=True)
    fig1.suptitle('Current'+light_label)
    for jtemp,jdirection,j in [(jx_data,'x',0),(jy_data,'y',1),(jz_data,'z',2)]:
        ax1[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax1[j].yaxis.major.formatter._useMathText = True 
        ax1[0].set_ylabel('$j^{tot}(t)$ A/cm$^2$')
        ax1[j].set_title(jdirection)
        ax1[j].set_xlabel('t (fs)')  
        ax1[j].plot(jtemp[:,0]/fs, jtemp[:,1])#, label='$j'+jdirection+'^{tot}(t)$    polarization = '+light_label)
    fig1.tight_layout()
    fig1.savefig(current_plot_output)
    plt.close(fig1)