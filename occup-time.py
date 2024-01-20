#! /usr/bin/env python
"""_summary_
This script plot the occupation functions with time, f(E,t). It also plots the E-axis maximum for the time derivative of f(E,t), (df/dt)max_in_E_axis.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as sgl
from scipy.optimize import curve_fit
import glob
import configparser
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('DMDana.ini')
Input=config['occup-time']

fs  = 41.341373335
sec = 4.1341373335E+16
Hatree_to_eV = 27.211386245988

timestep_for_all_files=Input.getint('timestep_for_occupation_output') #fs
t_tot = Input.getint('t_max') # fs
filelist_step=Input.getint('filelist_step') # select parts of the filelist

timestep_for_selected_file=timestep_for_all_files*filelist_step/1000 #ps
# Read all the occupations file names at once
occup_scat_files = glob.glob('occupations_t0.out')+sorted(glob.glob('occupations-*out'))
# Select partial files
occup_scat_files=occup_scat_files[::filelist_step]
# Read and plot data from one file at time 
fig = plt.figure(figsize=(12, 10))
ax =fig.add_subplot(111,projection='3d')
for index, file in enumerate(occup_scat_files):
    if index*timestep_for_selected_file>t_tot/1000:
        break
    data = np.loadtxt(file)
    ax.scatter(data[:,0]*Hatree_to_eV, data[:,1], zs=index*timestep_for_selected_file, zdir='y')
ax.set_xlim(0.05*Hatree_to_eV, 0.06*Hatree_to_eV)
ax.set_ylim(0, t_tot/1000)
ax.set_zlim(0, 1)

ax.set_xlabel('E (eV)')
ax.set_ylabel('t (ps)')
ax.set_zlabel('f')
ax.xaxis.set_pane_color('w')
ax.yaxis.set_pane_color('w')
ax.zaxis.set_pane_color('w')

ax.view_init(elev=20., azim=-70, roll=0)
fig.savefig('occup_time_Ttot%d_Step%d.png'%(t_tot,timestep_for_selected_file*1000), bbox_inches="tight")

