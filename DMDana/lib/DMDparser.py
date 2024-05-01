import configparser
import datetime
import glob
import logging
import os
from dataclasses import dataclass, field
from typing import List

import git
import numpy as np
from pydantic import BaseModel

from .. import do as DMDdo
from . import constant as const
from .param_in_class import param_class

libpath='/'.join(__file__.split('/')[0:-3])  # The path where DMDana is installed (not including DMDana)
allfuncname=['FFT_DC_convergence_test','current_plot','FFT_spectrum_plot','occup_time','occup_deriv']

def check_and_get_path( filepath):
    assert os.path.isfile(filepath),"%s does not exist."%filepath
    return filepath

'''Usage
To get mu and temperature from ldbd_size.dat, in which the content is like:
9.50043414701698e-04 # T
0.00000000000000e+00  0.00000000000000e+00  0.00000000000000e+00 # muMin, muMax mu
just use: 
mu,temperature=read_text_from_file('ldbd_data/ldbd_size.dat',["mu","# T"],[2,0])
Remember to manually cenvert the string to the data type you want.
dtypelist could either be a type or a list of types
'''
def read_text_from_file(filepath,marklist,locationlist,stop_at_first_find,dtypelist=str,sep=None) -> List:
    assert len(marklist)==len(locationlist),"marklist and locationlist should have the same length."
    assert type(dtypelist)==type or type(dtypelist)==list, "dtypelist should be a type or a list of types."

    def loop(filepath,marklist,stop_at_first_find,locationlist):
        resultlist=[None]*len(marklist)
        with open(filepath) as f:
            for line in f:
                for i in range(len(marklist)):
                    if stop_at_first_find and None not in resultlist:
                        return resultlist
                    if marklist[i] not in line:
                        continue
                    if stop_at_first_find:
                        resultlist[i]=line.split(sep)[locationlist[i]] if resultlist[i]==None else resultlist[i]
                    else:
                        resultlist[i]=line.split(sep)[locationlist[i]]
        return resultlist
    resultlist=loop(filepath,marklist,stop_at_first_find,locationlist)
    if type(dtypelist)==type:
        resultlist=[dtypelist(i) if i!=None else None for i in resultlist]
    elif type(dtypelist)==list:
        assert len(dtypelist)==len(resultlist), "dtypelist and resultlist should have the same length."
        resultlist=[dtypelist[i](resultlist[i]) if resultlist[i]!=None else None for i in range(len(resultlist))]
    return resultlist

def glob_occupation_files(folder):
    #assert glob.glob(folder+'/occupations_t0.out')!=[], "Did not found occupations_t0.out at folder %s"%folder
    occup_files = glob.glob(folder+'/occupations_t0.out')+sorted(glob.glob(folder+'/occupations-[0-9][0-9][0-9][0-9][0-9].out'))
    return occup_files

def get_total_step_number(folder):
    linenumber=0
    assert os.path.isfile(folder+'/jx_elec_tot.out'), "jx_elec_tot.out not found in %s"%folder
    with open(folder+'/jx_elec_tot.out') as f:
        for line in f:
            linenumber+=1
    return linenumber-3 # not include the step at t=0

def get_current_data(folder, elec_or_hole):
    """
    Loads current data from specified files based on particle type (electron, hole, or total).

    Args:
    folder (str): Directory path where the data files are stored.
    elec_or_hole (str): Type of particles - 'elec' for electrons, 'hole' for holes, or 'total' for combined data.

    Returns:
    tuple: Three numpy.ndarrays corresponding to the 'x', 'y', and 'z' components of the data.
    """

    def load_data(folder, particle_type, component):
        """
        Helper function to load data from a file based on folder path, particle type, and component.

        Args:
        folder (str): Directory path where the data files are stored.
        particle_type (str): 'elec' or 'hole' indicating the type of particle.
        component (str): 'x', 'y', or 'z' indicating the data component.

        Returns:
        numpy.ndarray: Array of data loaded from the specified file.
        """
        path = f"{folder}/j{component}_{particle_type}_tot.out"
        assert os.path.isfile(path), f"File not found: {path}"
        return np.loadtxt(path, skiprows=1)

    # Ensure the input type is one of the allowed options
    assert elec_or_hole in ['elec', 'hole', 'total'], "Invalid type specified"

    if elec_or_hole == 'total':
        # Load and sum data for both electrons and holes when 'total' is requested
        data_elec = {comp: load_data(folder, 'elec', comp) for comp in ['x', 'y', 'z']}
        data_hole = {comp: load_data(folder, 'hole', comp) for comp in ['x', 'y', 'z']}

        data = {}
        for comp in ['x', 'y', 'z']:
            # Combine electron and hole data by summing the second column of arrays
            data_comp = data_elec[comp].copy()
            data_comp[:, 1:] += data_hole[comp][:, 1:]
            data[comp] = data_comp
    else:
        # Load data for a single particle type (electron or hole)
        data = {comp: load_data(folder, elec_or_hole, comp) for comp in ['x', 'y', 'z']}

    return data['x'], data['y'], data['z']


def get_DMD_param(path='.'):
    """
    get the parameters in param.in

    Parameters
    1. path: the folder containing param.in

    Return
    1. DMDparam_value: a dictionary containing the parameters in param.in
    """
    DMDparam_value=dict()
    filepath = check_and_get_path(path+'/param.in')
    with open(filepath) as f:
        for line in f:
            line=line.strip()
            if line=='':
                continue
            elif line[0]=='#':
                continue
            line=line.split('#')[0]
            assert '=' in line, "param.in is not correctly setted."
            list_for_this_line=line.split('=')
            assert len(list_for_this_line)==2,"param.in is not correctly setted."
            DMDparam_value[list_for_this_line[0].strip()]=list_for_this_line[1].strip()
    return DMDparam_value

def get_mu_temperature(DMDparam_value,path='.'):
    """
    get mu and temperature from ldbd_data/ldbd_size.dat, param.in, and out(DMD.out)
    ldbd_data/ldbd_size.dat and out(DMD.out) are read from files
    content of param.in is read from the parameter "DMDparam_value"

    Parameters
    1. DMDparam_value: a dictionary containing the parameters in param.in
    2. path: the folder containing these files

    Return
    1. mu_au
    2. temperature_au
    """
    # read ldbd_size.dat
    filepath = check_and_get_path(path+'/ldbd_data/ldbd_size.dat')
    mu_au_text,temperature_au_text=read_text_from_file(filepath,marklist=["mu","# T"],locationlist=[2,0],stop_at_first_find=True)
    assert mu_au_text != None,"mu not found in ldbd_size.dat"
    assert temperature_au_text != None, "temperature not found in ldbd_size.dat"
    mu_au=float(mu_au_text)
    temperature_au=float(temperature_au_text)
    # read parameter "mu" in paramm.in
    if 'mu' in DMDparam_value:
        mu_au=float(DMDparam_value['mu'])/const.Hatree_to_eV
    # read parameter "carrier_density" in paramm.in
    for _ in [0]:
        if 'carrier_density' not in DMDparam_value:
            break
        if DMDparam_value['carrier_density'] ==None: 
            break
        if float(DMDparam_value['carrier_density'])==0:
            break
        assert os.path.isfile(path+'/out') or os.path.isfile(path+'/DMD.out'), "out or DMD.out file not found(for determine mu from non-zero carrier_density)"
        output_file_name=path+'/out' if os.path.isfile(path+'/out') else path+'/DMD.out'
        mu_au_text=read_text_from_file(output_file_name,marklist=["for given electron density"],locationlist=[5],stop_at_first_find=True)[0]
        assert mu_au_text != None, "carrier_density is not zero, but mu could not be found in out or DMD.out."
        mu_au=float(mu_au_text) 
    return mu_au,temperature_au

def get_erange(path='.'):
    """
    get different energy parameters from ldbd_data/ldbd_size.dat and ldbd_data/ldbd_erange_brange.dat

    Parameters
    1. path: the folder containing ldbd_data folder

    Return
    1.EBot_probe_au
    2.ETop_probe_au
    3.EBot_dm_au
    4.ETop_dm_au
    5.EBot_eph_au
    6.ETop_eph_au
    7.EvMax_au
    8.EcMin_au
    """
    filepath = check_and_get_path(path+'/ldbd_data/ldbd_size.dat')
    EBot_probe_au, ETop_probe_au, EBot_dm_au, ETop_dm_au, EBot_eph_au, ETop_eph_au=read_text_from_file(filepath,marklist=['# EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph']*6,locationlist=range(6),stop_at_first_find=True)
    filepath= check_and_get_path(path+'/ldbd_data/ldbd_erange_brange.dat')
    with open(filepath) as f:
        line=f.readline().split()
        assert len(line)==2, "The first line of ldbd_erange_brange.dat is not correctly setted."
        EvMax_au,EcMin_au=line
    return [float(i) for i in [EBot_probe_au, ETop_probe_au, EBot_dm_au, ETop_dm_au, EBot_eph_au, ETop_eph_au ,EvMax_au, EcMin_au]]



class default_class(BaseModel):
    only_jtot:bool
    folders:str
    def __getitem__(self,key):#For compatibility with configparser
        return str(getattr(self,key))
    def __setitem__(self,key,value_str):#For compatibility with configparser
        type_key=type(getattr(self,key))
        setattr(self,key,type_key(value_str))
    def getint(self,key):
        return int(getattr(self,key))
    def getfloat(self,key):
        return float(getattr(self,key))
    def getboolean(self,key):
        return bool(getattr(self,key))
    get=__getitem__
class current_plot_class(default_class):
    current_plot_output:str
    t_min:float
    t_max:float
    smooth_on:bool
    smooth_method:str
    smooth_times:int
    smooth_windowlen:int
    plot_all:bool
    elec_or_hole:str
    
class FFT_DC_convergence_test_class(default_class):
    Cutoff_step:int
    Cutoff_min:int
    Cutoff_max:int
    Window_type_list:str
    Database_output_csv:bool
    Database_output_xlsx:bool
    Database_output_filename_csv:str
    Database_output_filename_xlsx:str
    Figure_output_filename:str
    elec_or_hole:str
    
class FFT_spectrum_plot_class(default_class):
    Cutoff_list:int
    Window_type_list:str
    Log_y_scale:bool
    Summary_output_csv:bool
    Summary_output_xlsx:bool
    Summary_output_filename_csv:str
    Summary_output_filename_xlsx:str
    elec_or_hole:str
    
class occup_time_class(default_class):
    t_max:int
    filelist_step:int
    occup_time_plot_set_Erange:bool
    occup_time_plot_lowE:float
    occup_time_plot_highE:float
    plot_conduction_valence:bool
    plot_occupation_number_setlimit:bool
    plot_occupation_number_min:float
    plot_occupation_number_max:float
    output_all_figure_types:bool
    figure_style:str
    fit_Boltzmann:bool
    fit_Boltzmann_initial_guess_mu:float
    fit_Boltzmann_initial_guess_mu_auto:bool
    fit_Boltzmann_initial_guess_T:float
    fit_Boltzmann_initial_guess_T_auto:bool
    Substract_initial_occupation:bool
    showlegend:bool
class occup_deriv_class(default_class):
    t_max:int
    filelist_step:int
    
class DMDana_ini_config_setting_class(BaseModel):
   
    section_DEFAULT:default_class
    section_current_plot:current_plot_class
    section_FFT_DC_convergence_test:FFT_DC_convergence_test_class
    section_FFT_spectrum_plot:FFT_spectrum_plot_class
    section_occup_time:occup_time_class
    section_occup_deriv:occup_deriv_class    
    def __getitem__(self,key):
        return getattr(self,'section_'+key.replace('-','_'))
    #def __setitem__(self,key,value):
        #setattr(self,key,value)
    def items(self,key):
        return dict((key,str(val))for key,val in self[key].model_dump().items())
    
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
        self._configfile_path=value
        if value==None:
            return
        assert os.path.isfile(value), "%s does not exist"%value
        self.load_config_setting_from_file()
        for section_key,section_data in self.configsetting:
            section_data.folders=self.DMD_folder

        
    #Class and methods
    def __repr__(self) -> str:
        return __class__.__qualname__+'(configfile_path=%s)'%self._configfile_path
    
    def load_config_setting_from_file(self):
        self.DMDana_ini_configparser0 = configparser.ConfigParser(inline_comment_prefixes="#")
        if (os.path.isfile(self.configfile_path)):
            default_ini=check_and_get_path(libpath+'/DMDana/do/DMDana_default.ini')
            self.DMDana_ini_configparser0.read([default_ini,self.configfile_path])
        else:
            raise Warning('%s not exist. Default setting would be used. You could run "DMDana init" to initialize it.'%self.configfile_path)
        self.configsetting=DMDana_ini_config_setting_class(**dict( ('section_'+key.replace('-','_'),val)for key,val in self.DMDana_ini_configparser0.items()))# Use a new class to replace the configparser class
        #self.DMDana_ini_configparser=self.configsetting 
        # for compatibility with old version. 
        # This class support both dataclass structure(For new codes) and configparser structure(For compatibility with old codes).

    def FFT_DC_convergence_test(self):
        DMDdo.FFT_DC_convergence_test.do(Extented_Support(self))
    def FFT_spectrum_plot(self):
        DMDdo.FFT_spectrum_plot.do(Extented_Support(self))
    def current_plot(self):
        DMDdo.current_plot.do(Extented_Support(self))
    def occup_deriv(self):
        DMDdo.occup_deriv.do(Extented_Support(self))
    def occup_time(self):
        DMDdo.occup_time.do(Extented_Support(self))
    def get_config_result(self,funcname:str,show_init_log=True,folder_for_analysis_module_to_check=None):
        #func
        if folder_for_analysis_module_to_check==None:
            folder_for_analysis_module_to_check=self.DMD_folder
        assert funcname in allfuncname,'funcname is not correct.'
        if funcname in ['FFT_DC_convergence_test','current_plot','FFT_spectrum_plot']:
            config=config_current(funcname,self.DMDana_ini_configparser,folder_for_analysis_module_to_check,show_init_log=show_init_log)
        if funcname in ['occup_time','occup_deriv']:
            config=config_occup(funcname,self.DMDana_ini_configparser,folder_for_analysis_module_to_check,show_init_log=show_init_log)
        return config

class Extented_Support(analyze_class):
    #This is for the interactivity with the analysis modules, which helps to support some extended features, like plot for multiple folders together.
    def __init__(self,object:analyze_class):
        super().__init__(object.DMD_folder)
        self.configfile_path=str(object.configfile_path)
        self.configsetting=object.configsetting

    @property
    def configfile_path(self):
        return self._configfile_path
    @configfile_path.setter
    def configfile_path(self,val):
        super(Extented_Support,type(self)).configfile_path.fset(self,val)
        if val==None:
            return
        self.folderlist=self.configsetting.section_DEFAULT.folders.split(',')
    def set_multiple_folders(self,folderlist):
        self.folderlist=folderlist
        for section_key,section_data in self.configsetting.model_fields.items():
            section_data.folders=','.join(folderlist)
    def get_folder_name_by_number(self,folder_number):
        assert folder_number<len(self.folderlist) and folder_number>=0
        return self.folderlist[folder_number]
    def get_folder_config_result(self,funcname: str,folder_number: int,show_init_log=True):
        folder_for_analysis_module_to_check=self.get_folder_name_by_number(folder_number)
        return super().get_config_result(funcname,show_init_log,folder_for_analysis_module_to_check)
    get_folder_config=get_folder_config_result# for compatibility with old codes
    
    
    ## quicker set/get methods for operating the config settings.
    #def set(self,section: str,key: str,value):
    #    '''set options in DMDana.ini configparser structure.
    #    If you want to save int value, make sure explicitly convert it to int before using this function
    #    Or later reading process would report an error'''
    #    self.DMDana_ini_configparser[section][key]=str(value)
    #def get(self,section: str,key: str):
    #    return self.DMDana_ini_configparser[section][key]

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
    

class config_base(object): 
    def __init__(self, funcname_in,DMDana_ini_configparser: configparser.ConfigParser,folder='.',show_init_log=True):
        self.logfile=None
        self.DMDana_ini_configparser=DMDana_ini_configparser
        self.funcname=funcname_in
        self.EBot_probe_au=None
        self.ETop_probe_au=None 
        self.EBot_dm_au=None 
        self.ETop_dm_au=None 
        self.EBot_eph_au=None 
        self.ETop_eph_au=None
        self.EvMax_au=None
        self.EcMin_au=None
        self.mu_au=None
        self.temperature_au=None
        self.Input=self.DMDana_ini_configparser[self.funcname.replace('_','-')]# Due to historical reason, the name in DMDana.ini is with "-" instead of "_". Other parts of the code all use "_" for funcname
        if show_init_log==True:
            self.initiallog(self.funcname)
        self.folder=folder
        self.DMDparam_value=get_DMD_param(self.folder)# Use the param.in in the first folder

    def initiallog(self,funcname):#this should be done after setting global variable "funcname" 
        if os.path.isdir(libpath+'/.git'):# if the code is in develop mode (without pip install)
            repo = git.Repo(libpath,search_parent_directories=True)
            sha = repo.head.object.hexsha
        else:# if the code is in installed mode (with pip install)
            with open(libpath+'/DMDana/githash.log') as file:
                sha=file.readline().strip()
        logging.info("============DMDana============")
        logging.info("Git hash %s (%s)"%(sha[:7],sha))
        logging.info("Submodule: %s"%funcname)
        logging.info("Start time: %s"%datetime.datetime.now())
        logging.info("===Configuration Parameter===")
        paramdict=dict((self.DMDana_ini_configparser.items(funcname.replace('_','-'))))
        for i in paramdict:
            logging.info("%-35s"%i+':\t'+paramdict[i]+'')
        logging.info("===Initialization finished===")
        
class config_current(config_base):
    def __init__(self, funcname_in,DMDana_ini_configparser,folder='.',show_init_log=True):
        super().__init__(funcname_in,DMDana_ini_configparser,folder,show_init_log)
        self.only_jtot=None
        self.jx_data=None
        self.jy_data=None
        self.jz_data=None
        self.jx_data_path=None
        self.jy_data_path=None
        self.jz_data_path=None
        self.only_jtot=self.Input.getboolean('only_jtot')
        self.elec_or_hole=self.Input.get('elec_or_hole')
        assert self.only_jtot!=None, 'only_jtot is not correct setted.'
        self.jx_data,self.jy_data,self.jz_data=get_current_data(self.folder,self.elec_or_hole)
        assert len(self.jx_data)==len(self.jy_data)==len(self.jz_data), 'The line number in jx_data jy_data jz_data are not the same. Please deal with your data.'
        pumpPoltype=self.DMDparam_value['pumpPoltype']
        pumpA0=float(self.DMDparam_value['pumpA0'])
        pumpE=float(self.DMDparam_value['pumpE'])
        bandlabel={'elec':'conduction bands','hole':'valence bands','total':'all bands'}[self.elec_or_hole]
        self.light_label=' '+'for light of %s Polarization, %.2e a.u Amplitude, and %.2e eV Energy for %s'%(pumpPoltype,pumpA0,pumpE,bandlabel)
        
class config_occup(config_base):
    def __init__(self, funcname_in,DMDana_ini_configparser,folder='.',show_init_log=True):
        super().__init__(funcname_in,DMDana_ini_configparser,folder,show_init_log)
        self.occup_timestep_for_selected_file_fs=None
        self.occup_timestep_for_selected_file_ps=None
        self.occup_t_tot=None # maximum time to plot
        self.occup_maxmium_file_number_plotted_exclude_t0=None # maximum number of files to plot exclude t0
        self.occup_selected_files=None #The filelists to be dealt with. Note: its length might be larger than 
                                # occup_maxmium_file_number_plotted_exclude_t0+1 for later possible 
                                # calculations, eg. second-order finite difference
        self.mu_au,self.temperature_au=get_mu_temperature(self.DMDparam_value,path=self.folder)
        self.EBot_probe_au,self.ETop_probe_au,self.EBot_dm_au,self.ETop_dm_au,self.EBot_eph_au,self.ETop_eph_au,self.EvMax_au,self.EcMin_au=get_erange(path=self.folder)
        self.occup_selected_files = glob_occupation_files(self.folder)
        assert len(self.occup_selected_files)>=3, 'The number of occupation files is less than 3. Please check your data.'
        with open(self.occup_selected_files[1]) as f:
            firstline_this_file=f.readline()
            t1_fs=float(firstline_this_file.split()[12])/const.fs 
        with open(self.occup_selected_files[2]) as f:
            firstline_this_file=f.readline()
            t2_fs=float(firstline_this_file.split()[12])/const.fs
        data_first=np.loadtxt(self.occup_selected_files[0])
        try:
            self.occup_Emin_au=np.min(data_first[:,0])
            self.occup_Emax_au=np.max(data_first[:,0])
        except Exception as e:
            logging.error('%s file is in wrong format'%self.occup_selected_files[0])
            raise e
        self.occup_timestep_for_all_files=t2_fs-t1_fs #fs
        filelist_step=self.Input.getint('filelist_step') # select parts of the filelist
        self.occup_timestep_for_selected_file_fs=self.occup_timestep_for_all_files*filelist_step
        self.occup_timestep_for_selected_file_ps=self.occup_timestep_for_selected_file_fs/1000 #ps
        # Select partial files
        self.occup_selected_files=self.occup_selected_files[::filelist_step]
        self.occup_t_tot = self.Input.getfloat('t_max') # fs
        if self.occup_t_tot<=0:
            self.occup_maxmium_file_number_plotted_exclude_t0=len(self.occup_selected_files)-1
        else:
            self.occup_maxmium_file_number_plotted_exclude_t0=int(round(self.occup_t_tot/self.occup_timestep_for_selected_file_fs))
            assert self.occup_maxmium_file_number_plotted_exclude_t0<=len(self.occup_selected_files)-1 ,'occup_t_tot is larger than maximum time of data we have.'
            self.occup_selected_files=self.occup_selected_files[:self.occup_maxmium_file_number_plotted_exclude_t0+2]
            # keep the first few items of the filelists in need to avoid needless calculation cost.
            # If len(occup_selected_files)-1 > occup_maxmium_file_number_plotted_exclude_t0,
            # I Add one more points larger than points needed to be plotted for later possible second-order difference 
            # calculation. However if occup_maxmium_file_number_plotted_exclude_t0 happens to be len(occup_selected_files)-1, 
            # the number of points considered in later possible second-order difference calculation is still the number of points 
            # to be plotted
            # if in the future, more file number (more than occup_maxmium_file_number_plotted_exclude_t0+2) 
            # is needed to do calculation. This should be modified.
        self.occup_t_tot=self.occup_maxmium_file_number_plotted_exclude_t0*self.occup_timestep_for_selected_file_fs#fs
        
DMDana_ini_Class=Extented_Support# For compatibility with old version, use a new class to replace the old one

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
    