#!python
'''_summary_
This script plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.
(Using second center finite difference)
'''
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sgl
from scipy.optimize import curve_fit
import glob

fs  = 41.341373335
sec = 4.1341373335E+16
Hatree_to_eV = 27.2114
t_tot = 20.0
timestep_for_all_files=100 #fs
filelist_step=1 # select parts of the filelist
timestep_for_selected_file=timestep_for_all_files*filelist_step
# Read all the occupations file names at once
occup_scat_files = glob.glob('occupations_t0.out')+sorted(glob.glob('occupations-*out'))
# Select partial files
occup_scat_files=occup_scat_files[::filelist_step]


n = len(occup_scat_files)
data= np.full((n, 5646), np.nan)
dfdt = np.full((n,5646), np.nan)
dfdtMax = np.full((n), np.nan)
tarray=np.array(range(n))*timestep_for_selected_file/1000
for ind in range(n):
    data[ind,:] = np.loadtxt(occup_scat_files[ind])[:,1]
dfdt=np.gradient(data,timestep_for_selected_file,axis=0)
dfdtMax = np.max(dfdt,axis=1)
fig4, ax4 = plt.subplots(figsize=(10, 8))
ax4.plot(tarray, dfdtMax, '--.')
ax4.set_yscale('log')
ax4.set_xlabel('t (ps)')
ax4.set_ylabel(r'max$[\frac{df}{dt}]$')
fig4.savefig('dfdt-max.png')
