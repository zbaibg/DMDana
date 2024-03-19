#! /usr/bin/env python
"""_summary_
This plots the occupation functions with time, namely f(E,t)  of different styles in batch.
 
Be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from constant import *
from global_variable import config
from scipy.optimize import curve_fit
import logging
#Read input
occup_time_plot_lowE=config.Input.getfloat('occup_time_plot_lowE')
occup_time_plot_highE=config.Input.getfloat('occup_time_plot_highE')
occup_time_plot_lowE_conduction=config.Input.getfloat('occup_time_plot_lowE_conduction')
occup_time_plot_highE_conduction=config.Input.getfloat('occup_time_plot_highE_conduction')
occup_time_plot_lowE_valence=config.Input.getfloat('occup_time_plot_lowE_valence')
occup_time_plot_highE_valence=config.Input.getfloat('occup_time_plot_highE_valence')
occup_time_plot_set_Erange=config.Input.getboolean('occup_time_plot_set_Erange')
plot_occupation_number_min=config.Input.getfloat('plot_occupation_number_min')
plot_occupation_number_max=config.Input.getfloat('plot_occupation_number_max')
plot_conduction_valence=config.Input.getboolean('plot_conduction_valence')
Substract_initial_occupation=config.Input.getboolean('Substract_initial_occupation')
plot_occupation_number_setlimit=config.Input.getboolean("plot_occupation_number_setlimit")
fit_Boltzmann=config.Input.getboolean('fit_Boltzmann')
fit_Boltzmann_initial_guess_mu=config.Input.getfloat('fit_Boltzmann_initial_guess_mu')
fit_Boltzmann_initial_guess_mu_auto=config.Input.getboolean('fit_Boltzmann_initial_guess_mu_auto')
fit_Boltzmann_initial_guess_T=config.Input.getfloat('fit_Boltzmann_initial_guess_T')
fit_Boltzmann_initial_guess_T_auto=config.Input.getboolean('fit_Boltzmann_initial_guess_T_auto')
figure_style=config.Input.get('figure_style')
output_all_figure_types=config.Input.getboolean('output_all_figure_types')
occup_selected_files=config.occup_selected_files
occup_t_tot=config.occup_t_tot
occup_timestep_for_selected_file_ps=config.occup_timestep_for_selected_file_ps
#Hardcoded Setting 
cmap = mpl.colormaps['rainbow']
figsize=(12, 10)#Width, height in inches.
dpi=100.0#The resolution of the figure in dots-per-inch.
mu_au=config.mu_au
temperature_au=config.temperature_au
if fit_Boltzmann_initial_guess_mu_auto:
    fit_Boltzmann_initial_guess_mu=mu_au/eV
if fit_Boltzmann_initial_guess_T_auto:
    fit_Boltzmann_initial_guess_T=temperature_au/Kelvin
class plot_occup:
    def __init__(self,figure_style_this,Substract_initial_occupation_this):
        self.figure_style_this=figure_style_this #3D or heatmap
        self.Substract_initial_occupation_this=Substract_initial_occupation_this 
        self.fig = None
        if(self.figure_style_this not in ['3D','heatmap'] ):
            raise ValueError('figure_style should be 3D or heatmap')
        self.occupation_min_for_alldata=0#Must be 0, would be updated later
        self.occupation_max_for_alldata=0#Must be 0, would be updated later
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
    def fermi(self,temperature_au,mu_au,elist):
        ebyt = (elist - mu_au) / temperature_au
        occup=np.zeros(len(elist))
        occup[ebyt < -46]=1
        occup[ebyt > 46]=0
        index_middle=np.logical_and(ebyt >= -46, ebyt <= 46)
        occup[index_middle] = 1. / (np.exp(ebyt[index_middle]) + 1)
        return occup
    
    def read_occup_file(self,filename):
        with open(filename) as f:
            firstline_this_file=f.readline()
            time_this_file_fs=float(firstline_this_file.split()[12])/fs 
            #nstep=firstline.split()[9]
        data = np.loadtxt(filename)
        return time_this_file_fs,data
            
    def plot_one_file(self,time_this_file_fs,data):
        data=np.array(data)
        if self.Substract_initial_occupation_this:
            data[:,1]=data[:,1]-self.data_first[:,1]
        if occup_time_plot_set_Erange:
            #data=data[data[:,0].argsort()]
            data=data[np.logical_and(data[:,0]>occup_time_plot_lowE/Hatree_to_eV,data[:,0]<occup_time_plot_highE/Hatree_to_eV)]
        if fit_Boltzmann and occup_time_plot_lowE>=0 and self.Substract_initial_occupation_this==False:
            self.Boltzmann_fit_and_plot(data,time_this_file_fs)
        self.figtemp=self.plot_data_function(data,time_this_file_fs)            
        self.occupation_max_for_alldata=max(self.occupation_max_for_alldata,np.max(data[:,1]))
        self.occupation_min_for_alldata=min(self.occupation_min_for_alldata,np.min(data[:,1]))
    def plot_data(self,plot_data_function):
        self.plot_data_function=plot_data_function
        
        #Plot the fermi function at t=0
        data_temp=np.loadtxt('occupations_t0.out')
        self.data_first=np.array(data_temp)
        self.data_first[:,1]=self.fermi(temperature_au,mu_au,data_temp[:,0])
        self.plot_one_file(time_this_file_fs=0,data=self.data_first)
        #Plot all the other files
        for file in occup_selected_files:
            time_this_file_fs,data=self.read_occup_file(file)
            if time_this_file_fs>occup_t_tot:
                break            
            self.plot_one_file(time_this_file_fs,data)
    def Boltzmann_fit_and_plot(self,data,time_this_file_fs):
        popt,pcov=curve_fit(Bolzmann,data[:,0]*Hatree_to_eV,data[:,1],p0=[fit_Boltzmann_initial_guess_mu,fit_Boltzmann_initial_guess_T])
        logging.info("Boltzmann Distribution t(fs) %.3e mu(eV) %.3e T(K) %.3e"%(time_this_file_fs,popt[0],popt[1]))
        data_fit=np.zeros((1000,2))
        data_fit[:,0]=np.linspace(occup_time_plot_lowE/Hatree_to_eV,occup_time_plot_highE/Hatree_to_eV,1000)
        data_fit[:,1]=Bolzmann(data_fit[:,0]*Hatree_to_eV,*popt)
        self.plot_data_function(data_fit,time_this_file_fs,mode='scatter') 

    def plot_data_func_3D(self,data,time_this_file_fs,mode='scatter'):
        plot_param={'xs':data[:,0]*Hatree_to_eV,"ys":data[:,1], "zs":time_this_file_fs/1000, "zdir":'y',"c":[time_this_file_fs/1000]*data.shape[0],"cmap":cmap, "vmin":0,"vmax":occup_t_tot/1000}
        if mode=='scatter':
            return self.ax.scatter(**plot_param)
        elif mode=='plot':
            pass # not implemented yet
        
    def plot_data_func_heatmap(self,data,time_this_file_fs,mode='scatter'):
        plot_param={"x":data[:,0]*Hatree_to_eV, "y":data[:,1], "c":[time_this_file_fs/1000]*data.shape[0],"cmap":cmap, "vmin":0,"vmax":occup_t_tot/1000}
        if mode=='scatter':
            return self.ax.scatter(**plot_param)
        elif mode=='plot':
            pass # not implemented yet

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
        name="occup_time_Ttot%.1ffs_Step%.1ffs_%semin_%.1femax_%.1f.png"%(occup_t_tot,occup_timestep_for_selected_file_ps*1000,self.figure_style_this,occup_time_plot_lowE,occup_time_plot_highE)
        if self.Substract_initial_occupation_this:
            name="delta_"+name
        self.fig.savefig(name, bbox_inches="tight")   
        
def do():
    global occup_time_plot_set_Erange,occup_time_plot_highE,occup_time_plot_lowE,occup_time_plot_highE_conduction,occup_time_plot_lowE_conduction,occup_time_plot_highE_valence,occup_time_plot_lowE_valence
    logging.info('temperature(K): %.3e'%(temperature_au/Kelvin))
    logging.info('mu(eV): %.3e'%(mu_au/eV))

    do_sub()
    if plot_conduction_valence:
        occup_time_plot_set_Erange=True
        occup_time_plot_highE=occup_time_plot_highE_conduction
        occup_time_plot_lowE=occup_time_plot_lowE_conduction
        do_sub()
        occup_time_plot_highE=occup_time_plot_highE_valence
        occup_time_plot_lowE=occup_time_plot_lowE_valence
        do_sub()

def do_sub():
    if output_all_figure_types:
        for figure_style_each in ['3D','heatmap']:
            for Substract_initial_occupation_each in [True,False]:
                temp=plot_occup(figure_style_each,Substract_initial_occupation_each)
                temp.do()
    else:
        temp=plot_occup(figure_style,Substract_initial_occupation)
        temp.do()

#E mu in eV, T in Kelvin
def Bolzmann(E,mu,T):
    return np.exp(-(E-mu)*eV/(T*Kelvin))