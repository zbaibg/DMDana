import configparser
import os
from dataclasses import dataclass, field
from typing import List
import numpy as np
from DMDana.do.DMDana_ini_config_setting import DMDana_ini_config_setting_class
from DMDana.lib.DMDparser import check_and_get_path, get_DMD_param, get_erange, get_mu_temperature, get_total_step_number, glob_occupation_files, read_text_from_file
from ..do.config import (config_current, config_occup)
from .. import do as DMDdo
from . import constant as const
from .param_in_class import param_class
from .constant import libpath
class analyze_class:
    def __init__(self,DMD_folder):
        self.DMD_folder=DMD_folder
        self.configsetting:DMDana_ini_config_setting_class=None
        self.configfile_path:str=None
        self.config_result:config_results_class=config_results_class(self)
    @property
    def configsetting(self):
        return self._configsetting
    @configsetting.setter
    def configsetting(self,value):
        self._configsetting=value
    @property
    def DMDana_ini_configparser(self):
        return self._configsetting
    @DMDana_ini_configparser.setter
    def DMDana_ini_configparser(self,value):
        self._configsetting=value
    @property
    def configfile_path(self):
        return self._configfile_path
    
    @configfile_path.setter
    def configfile_path(self,value):
        #load DMDana.ini and overwrite the folder vallue with DMD_folder
        self._configfile_path=value
        if value==None:
            return
        assert os.path.isfile(value), "%s does not exist"%value
        self.__load_config_setting_from_file()
        for section_key,section_data in self.configsetting:
            section_data.folder=self.DMD_folder

    #Class and methods
    def __repr__(self) -> str:
        return __class__.__qualname__+'(configfile_path=%s)'%self._configfile_path
    

    def FFT_DC_convergence_test(self):
        DMDdo.FFT_DC_convergence_test.do(self)
    def FFT_spectrum_plot(self):
        DMDdo.FFT_spectrum_plot.do(self)
    def current_plot(self):
        DMDdo.current_plot.do(self)
    def occup_deriv(self):
        DMDdo.occup_deriv.do(self)
    def occup_time(self):
        DMDdo.occup_time.do(self)


@dataclass
class occupation_file_class:
    #Fields
    path:str
    
    data_eV:np.ndarray=field(init=False,repr=False)
    nstep:int=field(init=False)
    time_fs:float=field(init=False)
    #Initialization
    def __post_init__(self):
        # nk = 941 nb = 6 nstep = 1 t = 41.341374 tstep = 41.341374 tend = 1240241.225128
        self.nstep,time_au=read_text_from_file(self.path,['nk =']*2,[9,12],True,[int,float])
        self.time_fs=time_au/const.fs
        assert self.nstep!=None, "file %s might be empty"%self.path
        self.data_eV=np.loadtxt(self.path)
        self.data_eV[:,0]=self.data_eV[:,0]/const.eV
@dataclass
class occupations_class:
    #Fields
    DMD_folder:str
    list:List=field(init=False,repr=False)
    #Initialization
    def __post_init__(self):
        self.list=glob_occupation_files(self.DMD_folder)
    #Class and methods
    def file(self,i):
        return occupation_file_class(self.list[i])
@dataclass
class energy_class(object):
    #Field
    DMD_folder:str
    EBot_probe_eV:float=field(init=False)
    ETop_probe_eV:float=field(init=False)
    EBot_dm_eV:float=field(init=False)
    ETop_dm_eV:float=field(init=False)
    EBot_eph_eV:float=field(init=False)
    ETop_eph_eV:float=field(init=False)
    EvMax_eV:float=field(init=False)
    EcMin_eV:float=field(init=False)
    def __post_init__(self):
        self.EBot_probe_eV, self.ETop_probe_eV, self.EBot_dm_eV, self.ETop_dm_eV,\
        self.EBot_eph_eV, self.ETop_eph_eV ,self.EvMax_eV, self.EcMin_eV=\
        np.array(get_erange(self.DMD_folder))/const.eV
        
@dataclass
class lindblad_init_class:
    #Fields
    DMD_folder:str
    lindblad_folder:str=field(init=False)
    ldbd_data_folder:str=field(init=False)
    Full_k_mesh:List[int]=field(init=False)
    DFT_k_fold:List[int]=field(init=False)
    k_number:int=field(init=False)
    energy:energy_class=field(init=False)
    "nb nv bBot_dm bTop_dm bBot_eph(_elec) bTop_eph(_elec) nb_wannier bBot_probe band_dft_skipped"
    nb:int=field(init=False)
    nv:int=field(init=False)
    bBot_dm:int=field(init=False)
    bTop_dm:int=field(init=False)
    DM_Lower_E_eV:float=field(init=False)
    DM_Upper_E_eV:float=field(init=False)
    #bBot_eph_elec:int=field(init=False)
    #bTop_eph_elec:int=field(init=False)
    #nb_wannier:int=field(init=False)
    #bBot_probe:int=field(init=False)
    #band_dft_skipped:int=field(init=False)
    
    #Initialization
    def __post_init__(self):
        self.get_DMD_init_folder()
        self.energy=energy_class(DMD_folder=self.DMD_folder)
        self.get_kpoint_number()
        self.nb, self.nv, self.bBot_dm, self.bTop_dm=\
        read_text_from_file(self.ldbd_data_folder+'/ldbd_size.dat',marklist=['nb nv']*4,locationlist=[0,1,2,3],stop_at_first_find=True,dtypelist=int)
        self.DM_Lower_E_eV,self.DM_Upper_E_eV=read_text_from_file(self.lindblad_folder+'/lindbladInit.out',marklist=['Active energy range for density matrix']*2,locationlist=[6,8],stop_at_first_find=True,dtypelist=float)
    def get_DMD_init_folder(self):
        original_path=os.getcwd()
        os.chdir(self.DMD_folder+'/ldbd_data')
        self.ldbd_data_folder=os.getcwd()
        self.lindblad_folder=os.path.dirname(self.ldbd_data_folder)
        os.chdir(original_path)
    def get_kpoint_number(self):
        self.Full_k_mesh=read_text_from_file(self.lindblad_folder+'/lindbladInit.out',marklist=['Effective interpolated k-mesh dimensions']*3,locationlist=[5,6,7],stop_at_first_find=True,dtypelist=int)
        self.DFT_k_fold=read_text_from_file(self.lindblad_folder+'/lindbladInit.out',marklist=['kfold =']*3,locationlist=[3,4,5],stop_at_first_find=True,dtypelist=int)
        self.k_number=read_text_from_file(self.lindblad_folder+'/lindbladInit.out',marklist=['k-points with active states from'],locationlist=[1],stop_at_first_find=True,dtypelist=int)[0]

@dataclass
class DMD(object):
    #Fields
    DMD_folder:str
    param:param_class=field(init=False,repr=False)
    occupations:occupations_class=field(init=False,repr=False)
    analyze:analyze_class=field(init=False,repr=True)
    lindblad_init:lindblad_init_class=field(init=False,repr=False)
    total_time_fs:float=field(init=False,repr=False)
    total_step_num:int=field(init=False,repr=False)
    mu_eV:float=field(init=False,repr=False)
    temperature_K:float=field(init=False,repr=False)
    #Initialization
    def __post_init__(self):
        param_dist=get_DMD_param(self.DMD_folder)
        self.param=param_class(**param_dist)
        self.occupations=occupations_class(DMD_folder=self.DMD_folder)
        self.analyze=analyze_class(DMD_folder=self.DMD_folder)
        self.lindblad_init=lindblad_init_class(DMD_folder=self.DMD_folder)
        self.total_time_fs=None
        self.total_step_num=None
    def get_total_step_num_and_total_time_fs(self):
        self.total_step_num=get_total_step_number(self.DMD_folder)
        assert self.param.tstep_pump!=None, "tstep_pump is not setted in param.in"
        self.total_time_fs=self.total_step_num*self.param.tstep_pump
        return self.total_step_num,self.total_time_fs
    def start_analyze(self):
        self.analyze.configfile_path=self.DMD_folder+'/DMDana.ini'
    def get_mu_eV_and_T_K(self):
        mu_au,temperature_au=get_mu_temperature(self.param.model_dump(),self.DMD_folder)
        self.mu_eV,self.temperature_K=(mu_au/const.eV,temperature_au/const.Kelvin)
        return self.mu_eV,self.temperature_K
    
class config_results_class():
    def __init__(self,analyze_object):
        self.analyze:analyze_class=analyze_object
    @property
    def FFT_DC_convergence_test(self)->config_current:
        return self.analyze.get_config_result('FFT_DC_convergence_test',False)
    @property
    def FFT_spectrum_plot(self)->config_current:
        return self.analyze.get_config_result('FFT_spectrum_plot',False)
    @property
    def current_plot(self)->config_current:
        return self.analyze.get_config_result('current_plot',False)
    @property
    def occup_time(self)->config_occup:
        return self.analyze.get_config_result('occup_time',False)
    @property
    def occup_deriv(self)->config_occup:
        return self.analyze.get_config_result('occup_deriv',False)