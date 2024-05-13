import logging
import os
from typing import Any, List, Optional, Union

import numpy as np
from pydantic import Field, computed_field
from pydantic.dataclasses import dataclass

from ..do import (FFT_DC_convergence_test, FFT_spectrum_plot, current_plot,
                  occup_deriv, occup_time)
from ..do.config import config_current, config_occup
from ..do.DMDana_ini_config_setting import (
    DMDana_ini_config_setting_class,
    change_folder_for_DMDana_ini_config_setting, get_DMDana_ini_config_setting,
    section_default_class)
from . import constant as const
from .constant import allfuncname, libpath
from .DMDparser import (check_and_get_path, get_DMD_param, get_erange,
                        get_mu_temperature, get_total_step_number,
                        glob_occupation_files, read_text_from_file)
from .param_in_class import param_class


@dataclass(config=dict(arbitrary_types_allowed=True))
class config_gen_class():
    DMDana_ini_config_setting: DMDana_ini_config_setting_class
    logger:Optional[logging.Logger]=None
    init_log:bool=True
    def __post_init__(self):
        self._param_dict={'DMDana_ini_config_setting':self.DMDana_ini_config_setting,'logger':self.logger,'init_log':self.init_log}
    @computed_field
    @property
    def FFT_DC_convergence_test(self)->FFT_DC_convergence_test.config_FFT_DC_convergence_test:
        return FFT_DC_convergence_test.config_FFT_DC_convergence_test(**self._param_dict)
    @computed_field
    @property
    def FFT_spectrum_plot(self)->FFT_spectrum_plot.config_FFT_spectrum_plot:
        return FFT_spectrum_plot.config_FFT_spectrum_plot(**self._param_dict)
    @computed_field
    @property
    def current_plot(self)->current_plot.config_current_plot:
        return current_plot.config_current_plot(**self._param_dict)
    @computed_field
    @property
    def occup_time(self)->occup_time.config_occup_time:
        return occup_time.config_occup_time(**self._param_dict)
    @computed_field
    @property
    def occup_deriv(self)->occup_deriv.config_occup_deriv:
        return occup_deriv.config_occup_deriv(**self._param_dict)
@dataclass(config=dict(arbitrary_types_allowed=True))
class analyze_class:
    configfile_path:str
    DMD_folder:str=None
    DMDana_ini_config_setting:DMDana_ini_config_setting_class= Field(init=False,repr=False, default=None)
    config_gen:config_gen_class=Field(init=False,repr=False, default=None)
    def __post_init__(self):
        self.DMDana_ini_config_setting = get_DMDana_ini_config_setting(self.configfile_path)
        if self.DMD_folder is not None:
            change_folder_for_DMDana_ini_config_setting(self.DMDana_ini_config_setting,self.DMD_folder)
        self.config_gen:config_gen_class=config_gen_class(DMDana_ini_config_setting=self.DMDana_ini_config_setting)

    def config_logger(self,logger:Union[logging.Logger,None]) -> None:
        assert self.config_gen is not None, "config_gen is not initiated."
        self.config_gen.logger=logger
        
    def config_init_log(self,init_log:bool) -> None:
        assert self.config_gen is not None, "config_gen is not initiated."
        self.config_gen.init_log=init_log
        
    #Class and methods
    def __repr__(self) -> str:
        return __class__.__qualname__+'(configfile_path=%s)'%self.configfile_path
    def FFT_DC_convergence_test(self,optional_config: FFT_DC_convergence_test.config_FFT_DC_convergence_test=None):
        if optional_config is None:
            FFT_DC_convergence_test.do(self.config_gen.FFT_DC_convergence_test)
        else:
            FFT_DC_convergence_test.do(optional_config)
    def FFT_spectrum_plot(self,optional_config: FFT_spectrum_plot.config_FFT_spectrum_plot=None):
        if optional_config is None:
            FFT_spectrum_plot.do(self.config_gen.FFT_spectrum_plot)
        else:
            FFT_spectrum_plot.do(optional_config)
    def current_plot(self,optional_config: current_plot.config_current_plot=None):
        if optional_config is None:
            current_plot.do(self.config_gen.current_plot)
        else:
            current_plot.do(optional_config)
    def occup_deriv(self,optional_config: occup_deriv.config_occup_deriv=None):
        if optional_config is None:
            occup_deriv.do(self.config_gen.occup_deriv)
        else:
            occup_deriv.do(optional_config)
    def occup_time(self,optional_config: occup_time.config_occup_time=None):
        if optional_config is None:
            occup_time.do(self.config_gen.occup_time)
        else:
            occup_time.do(optional_config)
    def perform(self, funcname: str):
        assert funcname in allfuncname, f'{funcname} is not correct.'
        assert getattr(self, funcname) is function, f'{funcname} is assigned to a non-function by accident, check the codes'
        getattr(self, funcname)()

        
@dataclass(config=dict(arbitrary_types_allowed=True))
class occupation_file_class:
    #Fields
    path:str
    data_eV:np.ndarray = Field(init=False, repr=False, default=None)
    nstep:int = Field(init=False, default=None)
    time_fs:float = Field(init=False, default=None)
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
    list:List=Field(init=False, repr=False, default=None)
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
    EBot_probe_eV:float=Field(init=False, default=None)
    ETop_probe_eV:float=Field(init=False, default=None)
    EBot_dm_eV:float=Field(init=False, default=None)
    ETop_dm_eV:float=Field(init=False, default=None)
    EBot_eph_eV:float=Field(init=False, default=None)
    ETop_eph_eV:float=Field(init=False, default=None)
    EvMax_eV:float=Field(init=False, default=None)
    EcMin_eV:float=Field(init=False, default=None)
    def __post_init__(self):
        self.EBot_probe_eV, self.ETop_probe_eV, self.EBot_dm_eV, self.ETop_dm_eV,\
        self.EBot_eph_eV, self.ETop_eph_eV ,self.EvMax_eV, self.EcMin_eV=\
        np.array(get_erange(self.DMD_folder))/const.eV
        
@dataclass
class lindblad_init_class:
    #Fields
    DMD_folder:str
    lindblad_folder:str=Field(init=False, default=None)
    ldbd_data_folder:str=Field(init=False, default=None)
    Full_k_mesh:List[int]=Field(init=False, default=None)
    DFT_k_fold:List[int]=Field(init=False, default=None)
    k_number:int=Field(init=False, default=None)
    energy:energy_class=Field(init=False, default=None)
    "nb nv bBot_dm bTop_dm bBot_eph(_elec) bTop_eph(_elec) nb_wannier bBot_probe band_dft_skipped"
    nb:int=Field(init=False, default=None)
    nv:int=Field(init=False, default=None)
    bBot_dm:int=Field(init=False, default=None)
    bTop_dm:int=Field(init=False, default=None)
    DM_Lower_E_eV:float=Field(init=False, default=None)
    DM_Upper_E_eV:float=Field(init=False, default=None)
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
class DMD:
    #Fields
    DMD_folder:str
    param:param_class=Field(init=False,repr=False, default=None)
    occupations:occupations_class=Field(init=False,repr=False, default=None)
    lindblad_init:lindblad_init_class=Field(init=False,repr=False, default=None)
    total_time_fs:float=Field(init=False,repr=False, default=None)
    total_step_num:int=Field(init=False,repr=False, default=None)
    mu_eV:float=Field(init=False,repr=False, default=None)
    temperature_K:float=Field(init=False,repr=False, default=None)
    _analyze:analyze_class=Field(init=False,repr=False, default=None)
    #Initialization
    def __post_init__(self):
        #param_dist=get_DMD_param(self.DMD_folder)
        #self.param=param_class(**param_dist)
        self.param=get_DMD_param(self.DMD_folder)
        self.occupations=occupations_class(DMD_folder=self.DMD_folder)
        self.lindblad_init=lindblad_init_class(DMD_folder=self.DMD_folder)
        self.total_time_fs=None
        self.total_step_num=None
    @computed_field
    @property
    def analyze(self)->analyze_class:
        assert self._analyze is not None, "self.analyze is not initiated. please run init_analyze method first."
        return self._analyze
    @analyze.setter
    def analyze(self,value):
        self._analyze=value
    def init_analyze(self,configfile_path:str):
        self.analyze=analyze_class(DMD_folder=self.DMD_folder,configfile_path=configfile_path)
    def is_analyze_initiated(self):
        return self._analyze is not None
    def get_total_step_num_and_total_time_fs(self):
        self.total_step_num=get_total_step_number(self.DMD_folder)
        assert self.param.tstep_pump!=None, "tstep_pump is not setted in param.in"
        self.total_time_fs=self.total_step_num*self.param.tstep_pump
        return self.total_step_num,self.total_time_fs
    def start_analyze(self):
        self.analyze.configfile_path=self.DMD_folder+'/DMDana.ini'
    def get_mu_eV_and_T_K(self):
        mu_au,temperature_au=get_mu_temperature(self.param.__dict__,self.DMD_folder)
        self.mu_eV,self.temperature_K=(mu_au/const.eV,temperature_au/const.Kelvin)
        return self.mu_eV,self.temperature_K
    
