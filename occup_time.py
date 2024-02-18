#! /usr/bin/env python
"""_summary_
This script plot the occupation functions with time, f(E,t).
This program does not read time from file content for now. It only count the file numbers to get time. 
So be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from constant import *
from global_variable import config
#Read input
occup_time_plot_lowE=config.Input.getfloat('occup_time_plot_lowE')
occup_time_plot_highE=config.Input.getfloat('occup_time_plot_highE')
occup_time_plot_set_Erange=config.Input.getboolean('occup_time_plot_set_Erange',False)
plot_occupation_number_min=config.Input.getfloat('plot_occupation_number_min',0)
plot_occupation_number_max=config.Input.getfloat('plot_occupation_number_max',1)
Substract_initial_occupation=config.Input.getboolean('Substract_initial_occupation',True)
plot_occupation_number_setlimit=config.Input.getboolean("plot_occupation_number_setlimit",True)
figure_style=config.Input.get('figure_style','3D')
output_all_figure_types=config.Input.getboolean('output_all_figure_types',True)
occup_selected_files=config.occup_selected_files
occup_t_tot=config.occup_t_tot
occup_timestep_for_selected_file_ps=config.occup_timestep_for_selected_file_ps
def do_per_figure_type(figure_style_this,Substract_initial_occupation_this):
    fig = plt.figure(figsize=(12, 10))
    cmap = mpl.colormaps['rainbow']
    if figure_style_this=='3D':
        ax =fig.add_subplot(111,projection='3d')
    elif figure_style_this=='heatmap':
        ax= fig.add_subplot(111)
    else:
        raise ValueError('figure_type should be 3D or heatmap')
    occupation_max_for_alldata=0
    occupation_min_for_alldata=0
    data_first=np.loadtxt('occupations_t0.out')
    #note that t in occupations_t0.out is actually not 0
    #we just take it as 0 here.
    for index, file in enumerate(occup_selected_files):
        with open(file) as f:
            firstline_this_file=f.readline()
            time_this_file_fs=float(firstline_this_file.split()[12])/fs 
            #nstep=firstline.split()[9]
        if time_this_file_fs>occup_t_tot:
            break
        data = np.loadtxt(file)
        if index==0:
            data_first=np.array(data)
        if Substract_initial_occupation_this:
            data[:,1]=data[:,1]-data_first[:,1]
        if occup_time_plot_set_Erange:
            #data=data[data[:,0].argsort()]
            data=data[np.logical_and(data[:,0]>occup_time_plot_lowE,data[:,0]<occup_time_plot_highE)]
        if figure_style_this=='3D':
            figtemp=ax.scatter(data[:,0]*Hatree_to_eV, data[:,1], zs=time_this_file_fs/1000, zdir='y',c=[time_this_file_fs/1000]*data.shape[0],cmap=cmap, vmin=0,vmax=occup_t_tot/1000)
        elif figure_style_this=='heatmap':
            figtemp=ax.scatter(data[:,0]*Hatree_to_eV, data[:,1], c=[time_this_file_fs/1000]*data.shape[0],cmap=cmap, vmin=0,vmax=occup_t_tot/1000)               
        occupation_max_for_alldata=max(occupation_max_for_alldata,np.max(data[:,1]))
        occupation_min_for_alldata=min(occupation_min_for_alldata,np.min(data[:,1]))
    if occup_time_plot_set_Erange:
        ax.set_xlim(occup_time_plot_lowE, occup_time_plot_highE)
    ax.set_xlabel('E (eV)')
    fig.colorbar(figtemp,label='Time (ps)')
    if figure_style_this=='3D':    
        ax.set_ylim(0, occup_t_tot/1000)
        if plot_occupation_number_setlimit:
            ax.set_zlim(plot_occupation_number_min, plot_occupation_number_max)
        else:
            ax.set_zlim(occupation_min_for_alldata, occupation_max_for_alldata)
        ax.set_ylabel('t (ps)')
        if Substract_initial_occupation_this:
            ax.set_zlabel('f-f(t=0)')
        else:
            ax.set_zlabel('f')
        ax.xaxis.set_pane_color('w')
        ax.yaxis.set_pane_color('w')
        ax.zaxis.set_pane_color('w')
        ax.view_init(elev=20., azim=-70, roll=0)
    if figure_style_this=='heatmap':
        if plot_occupation_number_setlimit:
            ax.set_ylim(plot_occupation_number_min, plot_occupation_number_max)
        else:
            ax.set_ylim(occupation_min_for_alldata, occupation_max_for_alldata)
        if Substract_initial_occupation_this:
            ax.set_ylabel('f-f(t=0)')
        else:
            ax.set_ylabel('f')
    if Substract_initial_occupation_this:
        fig.savefig('delta_occup_time_Ttot%dfs_Step%dfs_%s.png'%(occup_t_tot,occup_timestep_for_selected_file_ps*1000,figure_style_this), bbox_inches="tight")
    else:
        fig.savefig('occup_time_Ttot%dfs_Step%dfs_%s.png'%(occup_t_tot,occup_timestep_for_selected_file_ps*1000,figure_style_this), bbox_inches="tight")

def do():
    if output_all_figure_types:
        for figure_style_each in ['3D','heatmap']:
            for Substract_initial_occupation_each in [True,False]:
                do_per_figure_type(figure_style_each,Substract_initial_occupation_each)
    else:
        do_per_figure_type(figure_style,Substract_initial_occupation)
