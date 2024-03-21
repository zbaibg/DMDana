#! /usr/bin/env python
"""_summary_
This plots the occupation functions with time, namely f(E,t)  of different styles in batch.
 
Be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from ..lib.constant import *
from .config import config_occup
from scipy.optimize import curve_fit
import logging
from mpl_toolkits.mplot3d.axes3d import Axes3D
from typing import Union
#Read input
class param_class(object):
    def __init__(self,config: config_occup):
        #Hardcoded Setting 
        self.cmap = mpl.colormaps['rainbow']
        self.figsize=(12, 10)#Width, height in inches.
        self.dpi=100.0#The resolution of the figure in dots-per-inch.
        #Read from config
        self.occup_time_plot_lowE=config.Input.getfloat('occup_time_plot_lowE')
        self.occup_time_plot_highE=config.Input.getfloat('occup_time_plot_highE')
        self.occup_time_plot_lowE_conduction=config.EcMin_au*Hatree_to_eV
        self.occup_time_plot_highE_conduction=config.ETop_dm_au*Hatree_to_eV
        self.occup_time_plot_lowE_valence=config.EBot_dm_au*Hatree_to_eV
        self.occup_time_plot_highE_valence=config.EvMax_au*Hatree_to_eV
        self.occup_time_plot_set_Erange=config.Input.getboolean('occup_time_plot_set_Erange')
        self.plot_occupation_number_min=config.Input.getfloat('plot_occupation_number_min')
        self.plot_occupation_number_max=config.Input.getfloat('plot_occupation_number_max')
        self.plot_conduction_valence=config.Input.getboolean('plot_conduction_valence')
        self.Substract_initial_occupation=config.Input.getboolean('Substract_initial_occupation')
        self.plot_occupation_number_setlimit=config.Input.getboolean("plot_occupation_number_setlimit")
        self.fit_Boltzmann=config.Input.getboolean('fit_Boltzmann')
        self.fit_Boltzmann_initial_guess_mu=config.Input.getfloat('fit_Boltzmann_initial_guess_mu')
        self.fit_Boltzmann_initial_guess_mu_auto=config.Input.getboolean('fit_Boltzmann_initial_guess_mu_auto')
        self.fit_Boltzmann_initial_guess_T=config.Input.getfloat('fit_Boltzmann_initial_guess_T')
        self.fit_Boltzmann_initial_guess_T_auto=config.Input.getboolean('fit_Boltzmann_initial_guess_T_auto')
        self.figure_style=config.Input.get('figure_style')
        self.output_all_figure_types=config.Input.getboolean('output_all_figure_types')
        self.occup_selected_files=config.occup_selected_files
        self.occup_t_tot=config.occup_t_tot
        self.occup_timestep_for_selected_file_ps=config.occup_timestep_for_selected_file_ps
        self.mu_au=config.mu_au
        self.EBot_dm_au=config.EBot_dm_au
        self.ETop_dm_au=config.ETop_dm_au
        self.EcMin_au=config.EcMin_au
        self.EvMax_au=config.EvMax_au
        self.temperature_au=config.temperature_au
        if self.fit_Boltzmann_initial_guess_mu_auto:
            self.fit_Boltzmann_initial_guess_mu=self.mu_au/eV
        if self.fit_Boltzmann_initial_guess_T_auto:
            self.fit_Boltzmann_initial_guess_T=self.temperature_au/Kelvin

    
class occup_time(object):
    def __init__(self,param: param_class):
        self.param=param
        self.figure_style_this=self.param.figure_style #3D or heatmap
        self.Substract_initial_occupation_this=self.param.Substract_initial_occupation 
        self.fig = None
        if(self.figure_style_this not in ['3D','heatmap'] ):
            raise ValueError('figure_style should be 3D or heatmap')
        self.occupation_min_for_alldata=0#Must be 0, would be updated later
        self.occupation_max_for_alldata=0#Must be 0, would be updated later
        self.figtemp=None # A temp file used for create colorbar
        self.ax: Union[plt.Axes,Axes3D] =None #Axes
        self.fitted=False
    def do(self):
        self.fig = plt.figure(figsize=self.param.figsize,dpi=self.param.dpi)
        self.pre_processing()
        self.plot_data()
        self.post_processing()
        self.save_fig()


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

    def pre_processing(self):
        if self.figure_style_this=='3D':
            self.ax =self.fig.add_subplot(111,projection='3d')
        elif self.figure_style_this=='heatmap':
            self.ax= self.fig.add_subplot(111)

    def plot_data(self):
        #Plot the fermi function at t=0
        data_temp=np.loadtxt('occupations_t0.out')
        self.data_first=np.array(data_temp)
        self.data_first[:,1]=self.fermi(self.param.temperature_au,self.param.mu_au,data_temp[:,0])
        self.plot_one_curve(time_this_file_fs=0,data=self.data_first)
        #Plot all the other files
        for file in self.param.occup_selected_files:
            time_this_file_fs,data=self.read_occup_file(file)
            if time_this_file_fs>self.param.occup_t_tot:
                break            
            self.plot_one_curve(time_this_file_fs,data)

    def plot_one_curve(self,time_this_file_fs,data):
        data=np.array(data)
        if self.Substract_initial_occupation_this:
            data[:,1]=data[:,1]-self.data_first[:,1]
        if self.param.occup_time_plot_set_Erange:
            #data=data[data[:,0].argsort()]
            data=data[np.logical_and(data[:,0]>self.param.occup_time_plot_lowE/Hatree_to_eV,data[:,0]<self.param.occup_time_plot_highE/Hatree_to_eV)]
        for _ in [None]:#Fit Bolzmman
            if not self.param.fit_Boltzmann:
                break
            if self.Substract_initial_occupation_this:
                break
            if not self.param.occup_time_plot_lowE>=0:
                break
            self.Boltzmann_fit_and_plot(data,time_this_file_fs)
            self.fitted=True
        self.figtemp=self.plot_data_func(data,time_this_file_fs)            
        self.occupation_max_for_alldata=max(self.occupation_max_for_alldata,np.max(data[:,1]))
        self.occupation_min_for_alldata=min(self.occupation_min_for_alldata,np.min(data[:,1]))

    def Boltzmann_fit_and_plot(self,data,time_this_file_fs):
        try:
            popt,pcov=curve_fit(Bolzmann,data[:,0]*Hatree_to_eV,data[:,1],p0=[self.param.fit_Boltzmann_initial_guess_mu,self.param.fit_Boltzmann_initial_guess_T])
        except:
            logging.error("Boltzmann fit failed for t(fs) %.3e"%(time_this_file_fs))
            return
        logging.info("Boltzmann Distribution t(fs) %.3e mu(eV) %.3e T(K) %.3e"%(time_this_file_fs,popt[0],popt[1]))
        data_fit=np.zeros((1000,2))
        data_fit[:,0]=np.linspace(self.param.occup_time_plot_lowE/Hatree_to_eV,self.param.occup_time_plot_highE/Hatree_to_eV,1000)
        data_fit[:,1]=Bolzmann(data_fit[:,0]*Hatree_to_eV,*popt)
        self.plot_data_func(data_fit,time_this_file_fs,mode='scatter') 

    def plot_data_func(self,data: np.ndarray,time_this_file_fs,mode='scatter'):
        if self.figure_style_this=='3D':
            plot_param={'xs':data[:,0]*Hatree_to_eV,"ys":data[:,1], "zs":time_this_file_fs/1000, "zdir":'y',"c":[time_this_file_fs/1000]*data.shape[0],"cmap":self.param.cmap, "vmin":0,"vmax":self.param.occup_t_tot/1000}
        elif self.figure_style_this=='heatmap':
            plot_param={"x":data[:,0]*Hatree_to_eV, "y":data[:,1], "c":[time_this_file_fs/1000]*data.shape[0],"cmap":self.param.cmap, "vmin":0,"vmax":self.param.occup_t_tot/1000}
        if mode=='scatter':
            return self.ax.scatter(**plot_param)
        elif mode=='plot':
            pass # not implemented yet

    def post_processing(self):
        if self.param.occup_time_plot_set_Erange:
            assert not self.param.occup_time_plot_lowE > self.param.ETop_dm_au*Hatree_to_eV , "Erange for plot is out of range of the data"
            assert not self.param.occup_time_plot_highE < self.param.EBot_dm_au*Hatree_to_eV , "Erange for plot is out of range of the data"
            assert not (self.param.occup_time_plot_lowE > self.param.EvMax_au*Hatree_to_eV and self.param.occup_time_plot_highE < self.param.EcMin_au*Hatree_to_eV) , "Erange for plot is out of range of the data"
            self.ax.set_xlim(self.param.occup_time_plot_lowE, self.param.occup_time_plot_highE)
        self.ax.set_xlabel('E (eV)')
        self.fig.colorbar(self.figtemp,label='Time (ps)')   
        if self.figure_style_this=='3D':
             self.post_processing_3D_special()
        elif self.figure_style_this=='heatmap':
            self.post_processing_heatmap_special()

    def post_processing_3D_special(self):
        self.ax.set_ylim(0, self.param.occup_t_tot/1000)
        if self.param.plot_occupation_number_setlimit:
            self.ax.set_zlim(self.param.plot_occupation_number_min, self.param.plot_occupation_number_max)
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
        
    def post_processing_heatmap_special(self):

        if self.param.plot_occupation_number_setlimit:
            self.ax.set_ylim(self.param.plot_occupation_number_min, self.param.plot_occupation_number_max)
        else:
            self.ax.set_ylim(self.occupation_min_for_alldata, self.occupation_max_for_alldata) 
        if self.Substract_initial_occupation_this:
            self.ax.set_ylabel('f-f(t=0)')
        else:
            self.ax.set_ylabel('f')        
    def save_fig(self):
        name="occup_time_Ttot%.1ffs_Step%.1ffs_%s_Emin%.3feV_Emax%.3feV%s.png"%\
        (self.param.occup_t_tot,self.param.occup_timestep_for_selected_file_ps*1000,\
         self.figure_style_this,self.param.occup_time_plot_lowE,self.param.occup_time_plot_highE,'_fitted' if self.fitted else '')
        if self.Substract_initial_occupation_this:
            name="delta_"+name
        self.fig.savefig(name, bbox_inches="tight")   
        
def do(config):
    param=param_class(config)
    logging.info('temperature(K): %.3e'%(param.temperature_au/Kelvin))
    logging.info('mu(eV): %.3e'%(param.mu_au/eV))

    do_sub(param)
    if param.plot_conduction_valence:
        param.occup_time_plot_set_Erange=True
        param.occup_time_plot_highE=param.occup_time_plot_highE_conduction
        param.occup_time_plot_lowE=param.occup_time_plot_lowE_conduction
        do_sub(param)
        param.occup_time_plot_highE=param.occup_time_plot_highE_valence
        param.occup_time_plot_lowE=param.occup_time_plot_lowE_valence
        do_sub(param)

def do_sub(param: param_class):
    if param.output_all_figure_types:
        for figure_style_each in ['3D','heatmap']:
            param.figure_style=figure_style_each
            for Substract_initial_occupation_each in [True,False]:
                param.Substract_initial_occupation=Substract_initial_occupation_each
                plot_object=occup_time(param)
                plot_object.do()
    else:
        plot_object=occup_time(param)
        plot_object.do()

#E mu in eV, T in Kelvin
def Bolzmann(E,mu,T):
    return np.exp(-(E-mu)*eV/(T*Kelvin))