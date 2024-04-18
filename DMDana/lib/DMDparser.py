import numpy as np
from . import DMDparser
from  pydantic import BaseModel
import os
from . import constant as const
import glob
from dataclasses import dataclass,make_dataclass,field
from .. import do as DMDdo 
from typing import List
from .param_in_class import param_class
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
def read_text_from_file(filepath,marklist,locationlist,stop_at_first_find,dtypelist=str) -> List:
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
                        resultlist[i]=line.split()[locationlist[i]] if resultlist[i]==None else resultlist[i]
                    else:
                        resultlist[i]=line.split()[locationlist[i]]
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
def get_current_data(folder):
    """
    get the data of the current vs time from jx(y,z)_elec_tot.out

    Parameters
    1. path: the folder containing jx(y,z)_elec_tot.out

    Return
    the data of the current vs time for each direction.
    The first column is time(au), the second column is current(A/cm^2).
    1. jx_data
    2. jy_data
    3. jz_data
    """
    jxpath=folder+"/jx_elec_tot.out"
    jypath=folder+"/jy_elec_tot.out"
    jzpath=folder+"/jz_elec_tot.out"
    jx_data=np.loadtxt(jxpath,skiprows=1)
    jy_data=np.loadtxt(jypath,skiprows=1)
    jz_data=np.loadtxt(jzpath,skiprows=1)
    return jx_data,jy_data,jz_data

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
        if float(DMDparam_value['carrier_density'])==0:
            break
        assert os.path.isfile(path+'/out') or os.path.isfile(path+'/DMD.out'), "out or DMD.out file not found(for determine mu from non-zero carrier_density)"
        output_file_name=path+'/out' if os.path.isfile(path+'/out') else path+'/DMD.out'
        mu_au_text=read_text_from_file(output_file_name,marklist=["for given electron density"],locationlist=[5],stop_at_first_find=True)[0]
        mu_au=float(mu_au_text) if mu_au_text!=None else mu_au
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
class current_plot_class(default_class):
    current_plot_output:str
    t_min:float
    t_max:float
    smooth_on:bool
    smooth_method:str
    smooth_times:int
    smooth_windowlen:int
    plot_all:bool
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
class FFT_spectrum_plot_class(default_class):
    Cutoff_list:int
    Window_type_list:str
    Log_y_scale:bool
    Summary_output_csv:bool
    Summary_output_xlsx:bool
    Summary_output_filename_csv:str
    Summary_output_filename_xlsx:str
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
class occup_deriv_class(default_class):
    t_max:int
    filelist_step:int
    
class DMDana_ini_class(BaseModel):
   
    section_DEFAULT:default_class
    section_current_plot:current_plot_class
    section_FFT_DC_convergence_test:FFT_DC_convergence_test_class
    section_FFT_spectrum_plot:FFT_spectrum_plot_class
    section_occup_time:occup_time_class
    section_occup_deriv:occup_deriv_class    
class analyze_class:
    def __init__(self,folder:str):
        self.folder:str=folder
        self.config:DMDana_ini_class=None
        self._configfile_path:str=None
        self._DMDana_ini=None
        self._configparser=None
        #self.use_local_config()
    @property
    def configfile_path(self):
        return self._configfile_path
    @configfile_path.setter
    def configfile_path(self,value):
        assert os.path.isfile(value), "%s does not exist"%value
        self._configfile_path=value
        self._DMDana_ini=DMDdo.config.DMDana_ini_Class(self.configfile_path)
        self._configparser=self._DMDana_ini.DMDana_ini_configparser
        self._configparser['DEFAULT']['folder']=self.folder
        self._DMDana_ini.folderlist=[self.folder]
        self.config=DMDana_ini_class(**dict( ('section_'+key.replace('-','_'),val)for key,val in self._configparser.items())) 
    def set_configparser(self):
        for key_,val in self._configparser.items():
            keydash='section_'+key_.replace('-','_')
            for key,subval in val.items():
                self._configparser[key_][key]=self.config[keydash][key]
    def __repr__(self) -> str:
        return __class__.__qualname__+'(configfile_path=%s)'%self._configfile_path
    #Class and methods
    def FFT_DC_convergence_test(self):
        self.set_configparser()
        DMDdo.FFT_DC_convergence_test.do(self._DMDana_ini)
    def FFT_spectrum_plot(self):
        self.set_configparser()
        DMDdo.FFT_spectrum_plot.do(self._DMDana_ini)
    def current_plot(self):
        self.set_configparser()
        DMDdo.current_plot.do(self._DMDana_ini)
    def occup_deriv(self):
        self.set_configparser()
        DMDdo.occup_deriv.do(self._DMDana_ini)
    def occup_time(self):
        self.set_configparser()
        DMDdo.occup_time.do(self._DMDana_ini)
    def use_local_config(self):
        self.configfile_path=self.folder+'/DMDana.ini'

@dataclass
class occupation_file_class:
    #Fields
    path:str
    
    data_fs:np.ndarray=field(init=False,repr=False)
    nstep:int=field(init=False)
    time_fs:float=field(init=False)
    #Initialization
    def __post_init__(self):
        # nk = 941 nb = 6 nstep = 1 t = 41.341374 tstep = 41.341374 tend = 1240241.225128
        self.nstep,time_au=DMDparser.read_text_from_file(self.path,['nk =']*2,[9,12],True,[int,float])
        self.time_fs=time_au/const.fs
        assert self.nstep!=None, "file %s might be empty"%self.path
        self.data_fs=np.loadtxt(self.path)
        self.data_fs[:,0]=self.data_fs[:,0]/const.fs
@dataclass
class occupations_class:
    #Fields
    folder:str
    list:List=field(init=False,repr=False)
    #Initialization
    def __post_init__(self):
        self.list=DMDparser.glob_occupation_files(self.folder)
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
class lindblad_init:
    #Fields
    DMD_folder:str
    lindblad_folder:str=field(init=False)
    ldbd_data_folder:str=field(init=False)
    Full_k_mesh:List[int]=field(init=False)
    DFT_k_fold:List[int]=field(init=False)
    k_number:int=field(init=False)
    energy:energy_class=field(init=False)
    
    #Initialization
    def __post_init__(self):
        self.get_DMD_init_folder()
        self.energy=energy_class(DMD_folder=self.DMD_folder)
        self.get_kpoint_number()
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
    folder:str
    param:param_class=field(init=False,repr=False)
    occupations:occupations_class=field(init=False,repr=False)
    analyze:analyze_class=field(init=False,repr=True)
    lindblad_init:lindblad_init=field(init=False,repr=False)
    total_time_fs:float=field(init=False,repr=False)
    total_step_num:int=field(init=False,repr=False)
    #Initialization
    def __post_init__(self):
        param_dist=DMDparser.get_DMD_param(self.folder)
        self.param=param_class(**param_dist)
        self.occupations=occupations_class(folder=self.folder)
        self.analyze=analyze_class(folder=self.folder)
        self.lindblad_init=lindblad_init(DMD_folder=self.folder)
        self.total_time_fs=None
        self.total_step_num=None
    def get_total_step_num_and_time(self):
        self.total_step_num=get_total_step_number(self.folder)
        self.total_time_fs=self.total_step_num*self.param.tstep_laser