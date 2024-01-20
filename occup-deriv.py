#! /usr/bin/env python
"""_summary_
This script plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.
(Using second center finite difference).
This program does not read time from file content for now. It only count the file numbers to get time. 
So be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sgl
from scipy.optimize import curve_fit
import glob
import configparser
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('DMDana.ini')
Input=config['occup-deriv']

fs  = 41.341373335
sec = 4.1341373335E+16
#Hatree_to_eV = 27.211386245988
timestep_for_all_files=Input.getint('timestep_for_occupation_output') #fs
filelist_step=Input.getint('filelist_step') # select parts of the filelist
timestep_for_selected_file=timestep_for_all_files*filelist_step#fs
# Read all the occupations file names at once
if glob.glob('occupations_t0.out')==[]:
    raise ValueError("Did not found occupations_t0.out")
occup_scat_files = glob.glob('occupations_t0.out')+sorted(glob.glob('occupations-*out'))
# Select partial files
occup_scat_files=occup_scat_files[::filelist_step]

t_tot = Input.getint('t_max') # fs
if t_tot<=0:
    maxmium_file_number_plotted_exclude_t0=len(occup_scat_files)-1
else:
    
    maxmium_file_number_plotted_exclude_t0=int(round(t_tot/timestep_for_selected_file))
    if maxmium_file_number_plotted_exclude_t0>len(occup_scat_files)-1:
        raise ValueError('t_tot is larger than maximum time of data we have.')
    occup_scat_files=occup_scat_files[:maxmium_file_number_plotted_exclude_t0+2]
    # keep the first few items of the filelists in need to avoid needless calculation cost.
    # If len(occup_scat_files)-1 > maxmium_file_number_plotted_exclude_t0.
    # I Add one more points larger than points needed to be plotted for later second-order difference 
    # calculation. However if maxmium_file_number_plotted_exclude_t0 happens to be len(occup_scat_files)-1, 
    # the number of points considered in second-order difference calculation is still the number of points 
    # to be plotted
t_tot=maxmium_file_number_plotted_exclude_t0*timestep_for_selected_file#fs
n = len(occup_scat_files)#number of files left in "occup_scat_files"
data= np.full((n, 5646), np.nan)# why 5646?
dfdt = np.full((n,5646), np.nan)
dfdtMax = np.full((n), np.nan)
tarray=np.array(range(n))*timestep_for_selected_file/1000#ps
for ind in range(n):
    data[ind,:] = np.loadtxt(occup_scat_files[ind])[:,1]
dfdt=np.gradient(data,timestep_for_selected_file,axis=0)#df/fs
dfdtMax = np.max(dfdt,axis=1)
fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.plot(tarray[0:maxmium_file_number_plotted_exclude_t0+1], dfdtMax[0:maxmium_file_number_plotted_exclude_t0+1], '--.')
ax4.set_yscale('log')
ax4.set_xlabel('t (ps)')
ax4.set_ylabel(r'max$[\frac{df}{dt}]$ unit:df/fs')
fig4.savefig('dfdt_max_Ttot%dfs_Step%dfs.png'%(t_tot,timestep_for_selected_file))
