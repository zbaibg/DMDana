#! /usr/bin/env python
"""
This plots the occupation functions with time, namely f(E,t)  of different styles in batch.
Be sure that your occupation filelists include complete number of files and also occupations_t0.out
"""
import logging
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.axes3d import Axes3D
from pydantic import Field
from pydantic.dataclasses import dataclass
from scipy.optimize import curve_fit

from ..lib import constant as const
from .config import DMDana_ini_config_setting_class, config_occup


@dataclass
class config_occup_time(config_occup):
    cmap: object = Field(default_factory=lambda: mpl.colormaps['rainbow'])
    figsize: tuple = (12, 10)  # Width, height in inches.
    dpi: float = 100.0  # The resolution of the figure in dots-per-inch.
    
    Substract_initial_occupation_this: bool = None
    figure_style_this: str = None
    occup_time_plot_set_Erange_this: bool = None
    occup_time_plot_lowE_this: float = None
    occup_time_plot_highE_this: Optional[float] = None
    data_fermi: object = None
    occup_time_plot_lowE_conduction: float = None
    occup_time_plot_highE_valence: float = None
    fit_Boltzmann_initial_guess_mu: float = None
    fit_Boltzmann_initial_guess_T: float = None
    def __post_init__(self):
        self.funcname='occup_time'
        super().__post_init__()
        if self.occup_time_plot_set_Erange_this is None:
            self.occup_time_plot_set_Erange_this=self.configsetting.occup_time_plot_set_Erange
        if self.occup_time_plot_set_Erange_this:
            if self.occup_time_plot_lowE_this is None:
                self.occup_time_plot_lowE_this=self.configsetting.occup_time_plot_lowE
            if self.occup_time_plot_highE_this is None:
                self.occup_time_plot_highE_this=self.configsetting.occup_time_plot_highE
        else:
            if self.occup_time_plot_lowE_this is None:
                self.occup_time_plot_lowE_this=self.occup_Emin_au*const.Hatree_to_eV
            if self.occup_time_plot_highE_this is None:
                self.occup_time_plot_highE_this=self.occup_Emax_au*const.Hatree_to_eV   
        if self.figure_style_this is None:
            self.figure_style_this=self.configsetting.figure_style
        if self.Substract_initial_occupation_this is None:
            self.Substract_initial_occupation_this=self.configsetting.Substract_initial_occupation
        if self.data_fermi is None:
            data_first=np.loadtxt(self.occup_selected_files[0])
            energylist=data_first[:,0]
            self.data_fermi=np.zeros(data_first.shape)
            self.data_fermi[:,0]=energylist
            self.data_fermi[:,1]=fermi(self.temperature_au,self.mu_au,energylist)  
        if self.occup_time_plot_lowE_conduction is None:
            self.occup_time_plot_lowE_conduction=max(self.EcMin_au*const.Hatree_to_eV, 0) - 0.01
        if self.occup_time_plot_highE_valence is None:
            self.occup_time_plot_highE_valence=min(self.EvMax_au*const.Hatree_to_eV, 0) + 0.01
        if self.configsetting.fit_Boltzmann_initial_guess_mu_auto:
            self.fit_Boltzmann_initial_guess_mu=self.EcMin_au/const.eV
        else:
            self.fit_Boltzmann_initial_guess_mu=self.configsetting.fit_Boltzmann_initial_guess_mu
        if self.configsetting.fit_Boltzmann_initial_guess_T_auto:
            self.fit_Boltzmann_initial_guess_T=self.temperature_au/const.Kelvin
        else:
            self.fit_Boltzmann_initial_guess_T=self.configsetting.fit_Boltzmann_initial_guess_T
def fermi(temperature_au,mu_au,elist):
    ebyt = (elist - mu_au) / temperature_au
    occup=np.zeros(len(elist))
    occup[ebyt < -46]=1
    occup[ebyt > 46]=0
    index_middle=np.logical_and(ebyt >= -46, ebyt <= 46)
    occup[index_middle] = 1. / (np.exp(ebyt[index_middle]) + 1)
    return occup

class plot_occup_time(object):
    def __init__(self,param: config_occup_time):
        self.param=param
        self.figure_style_this=self.param.figure_style_this #3D or heatmap
        self.Substract_initial_occupation_this=self.param.Substract_initial_occupation_this 
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
        plt.close()

    def read_occup_file(self,filename):
        try:
            with open(filename) as f:
                firstline_this_file=f.readline()
                time_this_file_fs=float(firstline_this_file.split()[12])/const.fs 
                #nstep=firstline.split()[9]
            data = np.loadtxt(filename)
            assert len(data)>0, 'the occupation file is empty'
        except Exception as e:
            logging.error('cannot correctly read occupation file: %s'%filename)
            raise e
        return time_this_file_fs,data

    def pre_processing(self):
        if self.figure_style_this=='3D':
            self.ax =self.fig.add_subplot(111,projection='3d')
        elif self.figure_style_this=='heatmap':
            self.ax= self.fig.add_subplot(111)

    def plot_data(self):
        #Plot the fermi function at t=0
        self.plot_one_curve(time_this_file_fs=0,data=self.param.data_fermi)
        #Plot all the other files
        for file in self.param.occup_selected_files:
            time_this_file_fs,data=self.read_occup_file(file)
            if time_this_file_fs>self.param.occup_t_tot+self.param.occup_timestep_for_selected_file_ps*1000/2:
                break            
            self.plot_one_curve(time_this_file_fs,data)

    def plot_one_curve(self,time_this_file_fs,data):    
        data=data.copy()
        if self.Substract_initial_occupation_this:
            data[:,1]=data[:,1]-self.param.data_fermi[:,1]
        data=data[np.logical_and(data[:,0]>self.param.occup_time_plot_lowE_this/const.Hatree_to_eV,data[:,0]<self.param.occup_time_plot_highE_this/const.Hatree_to_eV)]
        assert len(data)>0, 'No data found in the energy range (%.3e,%.3e)'%(self.param.occup_time_plot_lowE_this, self.param.occup_time_plot_highE_this)   
        for _ in [None]:#Fit Bolzmman
            if not self.param.configsetting.fit_Boltzmann:
                break
            if self.Substract_initial_occupation_this:
                break
            if not self.param.occup_time_plot_lowE_this>=self.param.occup_time_plot_lowE_conduction:
                break
            self.Boltzmann_fit_and_plot(data,time_this_file_fs)
            self.fitted=True
        self.figtemp=self.plot_data_func(data,time_this_file_fs) 
        self.occupation_max_for_alldata=max(self.occupation_max_for_alldata,np.max(data[:,1]))
        self.occupation_min_for_alldata=min(self.occupation_min_for_alldata,np.min(data[:,1]))

    def Boltzmann_fit_and_plot(self,data,time_this_file_fs):
        try:
            popt,pcov=curve_fit(Bolzmann,data[:,0]*const.Hatree_to_eV,data[:,1],p0=[self.param.fit_Boltzmann_initial_guess_mu,self.param.fit_Boltzmann_initial_guess_T])
        except:
            logging.error("Boltzmann fit failed for t(fs) %.3e"%(time_this_file_fs))
            return
        logging.info("Boltzmann Distribution t(fs) %.3e mu(eV) %.3e T(K) %.3e"%(time_this_file_fs,popt[0],popt[1]))
        data_fit=np.zeros((1000,2))
        data_fit[:,0]=np.linspace(self.param.occup_time_plot_lowE_this/const.Hatree_to_eV,self.param.occup_time_plot_highE_this/const.Hatree_to_eV,1000)
        data_fit[:,1]=Bolzmann(data_fit[:,0]*const.Hatree_to_eV,*popt)
        self.plot_data_func(data_fit,time_this_file_fs,mode='plot') 

    def plot_data_func(self,data: np.ndarray,time_this_file_fs,mode='scatter'):
        color=plt.cm.rainbow(time_this_file_fs/self.param.occup_t_tot)
        if mode=='scatter':
            if self.figure_style_this=='3D': 
                plot_param={'xs':data[:,0]*const.Hatree_to_eV,"ys":data[:,1], "zs":time_this_file_fs/1000, "zdir":'y',"c":[time_this_file_fs/1000]*data.shape[0],"cmap":self.param.cmap, "vmin":0,"vmax":self.param.occup_t_tot/1000}
                return self.ax.scatter(**plot_param,label='%.3e fs'%time_this_file_fs)
            elif self.figure_style_this=='heatmap' :
                plot_param={"x":data[:,0]*const.Hatree_to_eV, "y":data[:,1], "c":[time_this_file_fs/1000]*data.shape[0],"cmap":self.param.cmap, "vmin":0,"vmax":self.param.occup_t_tot/1000}
                return self.ax.scatter(**plot_param,label='%.3e fs'%time_this_file_fs)
        elif mode=='plot':
            if self.figure_style_this=='3D':
                plot_param={'xs':data[:,0]*const.Hatree_to_eV,"ys":data[:,1], "zs":time_this_file_fs/1000, "zdir":'y','c':color}
                return self.ax.plot(**plot_param,label='%.3e fs'%time_this_file_fs)
            elif self.figure_style_this=='heatmap':
                return self.ax.plot(data[:,0]*const.Hatree_to_eV, data[:,1],c=color,label='%.3e fs'%time_this_file_fs)

    def post_processing(self):
        assert not self.param.occup_time_plot_lowE_this > self.param.occup_Emax_au*const.Hatree_to_eV , "Erange for plot is out of range of the data"
        assert not self.param.occup_time_plot_highE_this < self.param.occup_Emin_au*const.Hatree_to_eV , "Erange for plot is out of range of the data"
        assert not (self.param.occup_time_plot_lowE_this > self.param.EvMax_au*const.Hatree_to_eV and self.param.occup_time_plot_highE_this < self.param.EcMin_au*const.Hatree_to_eV) , "Erange for plot is out of range of the data"
        self.ax.set_xlim(self.param.occup_time_plot_lowE_this, self.param.occup_time_plot_highE_this)
        self.ax.set_xlabel('E (eV)')
        self.fig.colorbar(self.figtemp,label='Time (ps)')   
        if self.figure_style_this=='3D':
             self.post_processing_3D_special()
        elif self.figure_style_this=='heatmap':
            self.post_processing_heatmap_special()
        if self.param.configsetting.showlegend:
            self.ax.legend()

    def post_processing_3D_special(self):
        self.ax.set_ylim(0, self.param.occup_t_tot/1000)
        if self.param.configsetting.plot_occupation_number_setlimit:
            self.ax.set_zlim(self.param.configsetting.plot_occupation_number_min, self.param.configsetting.plot_occupation_number_max)
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

        if self.param.configsetting.plot_occupation_number_setlimit:
            self.ax.set_ylim(self.param.configsetting.plot_occupation_number_min, self.param.configsetting.plot_occupation_number_max)
        else:
            self.ax.set_ylim(self.occupation_min_for_alldata, self.occupation_max_for_alldata) 
        if self.Substract_initial_occupation_this:
            self.ax.set_ylabel('f-f(t=0)')
        else:
            self.ax.set_ylabel('f')        
    def save_fig(self):
        name="occup_time_Ttot%.1ffs_Step%.1ffs_%s_Emin%.3feV_Emax%.3feV%s.png"%\
        (self.param.occup_t_tot,self.param.occup_timestep_for_selected_file_ps*1000,\
         self.figure_style_this,self.param.occup_time_plot_lowE_this,self.param.occup_time_plot_highE_this,'_fitted' if self.fitted else '')
        if self.Substract_initial_occupation_this:
            name="delta_"+name
        self.fig.savefig(name, bbox_inches="tight")   
        
def do(DMDana_ini_config_setting:DMDana_ini_config_setting_class):
    config=config_occup_time(DMDana_ini_config_setting=DMDana_ini_config_setting)
    logging.info('temperature(K): %.3e'%(config.temperature_au/const.Kelvin))
    logging.info('mu(eV): %.3e'%(config.mu_au/const.eV))

    do_sub(config)
    if config.configsetting.plot_conduction_valence:
        config.occup_time_plot_set_Erange_this=True
        config.occup_time_plot_highE_this=config.occup_Emax_au*const.Hatree_to_eV
        config.occup_time_plot_lowE_this=config.occup_time_plot_lowE_conduction
        do_sub(config)
        config.occup_time_plot_highE_this=config.occup_time_plot_highE_valence
        config.occup_time_plot_lowE_this=config.occup_Emin_au*const.Hatree_to_eV
        do_sub(config)

def do_sub(config: config_occup_time):
    if config.configsetting.output_all_figure_types:
        for figure_style_each in ['3D','heatmap']:
            config.figure_style_this=figure_style_each
            for Substract_initial_occupation_each in [True,False]:
                config.Substract_initial_occupation_this=Substract_initial_occupation_each
                plot_object=plot_occup_time(config)
                plot_object.do()
    else:
        plot_object=plot_occup_time(config)
        plot_object.do()

#E mu in eV, T in Kelvin
def Bolzmann(E,mu,T):
    return np.exp(-(E-mu)*const.eV/(T*const.Kelvin))