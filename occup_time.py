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
occup_time_plot_set_Erange=config.Input.getboolean('occup_time_plot_set_Erange')
plot_occupation_number_min=config.Input.getfloat('plot_occupation_number_min')
plot_occupation_number_max=config.Input.getfloat('plot_occupation_number_max')
Substract_initial_occupation=config.Input.getboolean('Substract_initial_occupation')
plot_occupation_number_setlimit=config.Input.getboolean("plot_occupation_number_setlimit")
figure_style=config.Input.get('figure_style')
output_all_figure_types=config.Input.getboolean('output_all_figure_types')
occup_selected_files=config.occup_selected_files
occup_t_tot=config.occup_t_tot
occup_timestep_for_selected_file_ps=config.occup_timestep_for_selected_file_ps
#Hardcoded Setting 
cmap = mpl.colormaps['rainbow']
figsize=(12, 10)#Width, height in inches.
dpi=100.0#The resolution of the figure in dots-per-inch.

class plot_occup:
    def __init__(self,figure_style_this,Substract_initial_occupation_this):
        self.figure_style_this=figure_style_this #3D or heatmap
        self.Substract_initial_occupation_this=Substract_initial_occupation_this 
        self.fig = None
        if(self.figure_style_this not in ['3D','heatmap'] ):
            raise ValueError('figure_style should be 3D or heatmap')
        self.occupation_min_for_alldata=None
        self.occupation_max_for_alldata=None
        self.figtemp=None # A temp file used for create colorbar
    def do(self):
        self.fig = plt.figure(figsize=figsize,dpi=dpi)
        if self.figure_style_this=='3D':
            self.ax =self.fig.add_subplot(111,projection='3d')
            self.plot_data(self.plot_data_func_3D)
            self.plot_post_processing_common()
            self.plot_post_processing_3D_special()
        elif self.figure_style_this=='heatmap':
            self.ax= self.fig.add_subplot(111)
            self.plot_data(self.plot_data_func_heatmap)
            self.plot_post_processing_common()
            self.plot_post_processing_heatmap_special()
        self.save_fig()
    def plot_post_processing_common(self):
        if occup_time_plot_set_Erange:
            self.ax.set_xlim(occup_time_plot_lowE, occup_time_plot_highE)
        self.ax.set_xlabel('E (eV)')
        self.fig.colorbar(self.figtemp,label='Time (ps)')    
    
    def plot_data(self,plot_data_function):
        temp_occupation_max_for_alldata=0
        temp_occupation_min_for_alldata=0
        #note that t in occupations_t0.out is actually not 0
        #we just take it as 0 here.
        data_first=np.loadtxt('occupations_t0.out')
        for index, file in enumerate(occup_selected_files):
            with open(file) as f:
                firstline_this_file=f.readline()
                time_this_file_fs=float(firstline_this_file.split()[12])/fs 
                #nstep=firstline.split()[9]
            if time_this_file_fs>occup_t_tot:
                break
            data = np.loadtxt(file)
            #if index==0:
                #data_first=np.array(data)
            if self.Substract_initial_occupation_this:
                data[:,1]=data[:,1]-data_first[:,1]
            if occup_time_plot_set_Erange:
                #data=data[data[:,0].argsort()]
                data=data[np.logical_and(data[:,0]>occup_time_plot_lowE,data[:,0]<occup_time_plot_highE)]
            figtemp=plot_data_function(data,time_this_file_fs)            
            temp_occupation_max_for_alldata=max(temp_occupation_max_for_alldata,np.max(data[:,1]))
            temp_occupation_min_for_alldata=min(temp_occupation_min_for_alldata,np.min(data[:,1]))
        self.occupation_min_for_alldata= temp_occupation_min_for_alldata
        self.occupation_max_for_alldata= temp_occupation_max_for_alldata 
        self.figtemp=figtemp   
    def plot_data_func_3D(self,data,time_this_file_fs):
        return self.ax.scatter(data[:,0]*Hatree_to_eV, data[:,1], zs=time_this_file_fs/1000, zdir='y',c=[time_this_file_fs/1000]*data.shape[0],cmap=cmap, vmin=0,vmax=occup_t_tot/1000)
    
    def plot_data_func_heatmap(self,data,time_this_file_fs):
        return self.ax.scatter(data[:,0]*Hatree_to_eV, data[:,1], c=[time_this_file_fs/1000]*data.shape[0],cmap=cmap, vmin=0,vmax=occup_t_tot/1000)   

    def plot_post_processing_3D_special(self):
        self.ax.set_ylim(0, occup_t_tot/1000)
        if plot_occupation_number_setlimit:
            self.ax.set_zlim(plot_occupation_number_min, plot_occupation_number_max)
        else:
            self.ax.set_zlim(self.occupation_min_for_alldata, self.occupation_max_for_alldata)          
        self.ax.set_ylabel('t (ps)')
        if self.Substract_initial_occupation_this:
            self.ax.set_zlabel('f-f(t=0)')
        else:
            self.ax.set_zlabel('f')
        self.ax.xaxis.set_pane_color('w')
        self.ax.yaxis.set_pane_color('w')
        self.ax.zaxis.set_pane_color('w')
        self.ax.view_init(elev=20., azim=-70, roll=0)  
        
    def plot_post_processing_heatmap_special(self):
        if plot_occupation_number_setlimit:
            self.ax.set_ylim(plot_occupation_number_min, plot_occupation_number_max)
        else:
            self.ax.set_ylim(self.occupation_min_for_alldata, self.occupation_max_for_alldata) 
        if self.Substract_initial_occupation_this:
            self.ax.set_ylabel('f-f(t=0)')
        else:
            self.ax.set_ylabel('f')        
    def save_fig(self):
        if self.Substract_initial_occupation_this:
            self.fig.savefig('delta_occup_time_Ttot%dfs_Step%dfs_%s.png'%(occup_t_tot,occup_timestep_for_selected_file_ps*1000,self.figure_style_this), bbox_inches="tight")
        else:
            self.fig.savefig('occup_time_Ttot%dfs_Step%dfs_%s.png'%(occup_t_tot,occup_timestep_for_selected_file_ps*1000,self.figure_style_this), bbox_inches="tight")   
        
def do():
    if output_all_figure_types:
        for figure_style_each in ['3D','heatmap']:
            for Substract_initial_occupation_each in [True,False]:
                temp=plot_occup(figure_style_each,Substract_initial_occupation_each)
                temp.do()
    else:
        temp=plot_occup(figure_style,Substract_initial_occupation)
        temp.do()
