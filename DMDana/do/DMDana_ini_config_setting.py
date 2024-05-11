import configparser
import os
from typing import Any, List, Union

from pydantic import BaseModel

from ..lib.constant import libpath
from ..lib.DMDparser import check_and_get_path


class section_default_class(BaseModel):
    only_jtot:bool
    folder:str
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
class sections_current_classes(section_default_class):#for all current sections
    elec_or_hole:str
class sections_occup_classes(section_default_class):#for all occup sections
    t_max:int
    filelist_step:int
class section_current_plot_class(sections_current_classes):
    current_plot_output:str
    t_min:float
    t_max:float
    smooth_on:bool
    smooth_method:str
    smooth_times:int
    smooth_windowlen:int
    plot_all:bool

    
class section_FFT_DC_convergence_test_class(sections_current_classes):
    Cutoff_step:int
    Cutoff_min:int
    Cutoff_max:int
    Window_type_list:Union[str,List[str]]
    Database_output_csv:bool
    Database_output_xlsx:bool
    Database_output_filename_csv:str
    Database_output_filename_xlsx:str
    Figure_output_filename:str
    def model_post_init(self,__context: Any):
        if isinstance(self.Window_type_list,str):
            self.Window_type_list=[i.strip() for i in self.Window_type_list.split(',')]  
    
class section_FFT_spectrum_plot_class(sections_current_classes):
    Cutoff_list:Union[str,List[int]]
    Window_type_list:Union[str,List[str]]
    Log_y_scale:bool
    Summary_output_csv:bool
    Summary_output_xlsx:bool
    Summary_output_filename_csv:str
    Summary_output_filename_xlsx:str
    def model_post_init(self,__context: Any):
        if isinstance(self.Window_type_list,str):
            self.Window_type_list=[i.strip() for i in self.Window_type_list.split(',')]  
        if isinstance(self.Cutoff_list,str):
            self.Cutoff_list=[int(i) for i in self.Cutoff_list.split(',')]
class section_occup_time_class(sections_occup_classes):
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
class section_occup_deriv_class(sections_occup_classes):
    pass
    
class DMDana_ini_config_setting_class(BaseModel):
   
    section_DEFAULT:section_default_class
    section_current_plot:section_current_plot_class
    section_FFT_DC_convergence_test:section_FFT_DC_convergence_test_class
    section_FFT_spectrum_plot:section_FFT_spectrum_plot_class
    section_occup_time:section_occup_time_class
    section_occup_deriv:section_occup_deriv_class    

    def __getitem__(self,key):
        return getattr(self,'section_'+key.replace('-','_'))
    #def __setitem__(self,key,value):
        #setattr(self,key,value)
    def items(self,key):
        return dict((key,str(val))for key,val in self[key].model_dump().items())
    

def get_DMDana_ini_config_setting(configfile_path)->DMDana_ini_config_setting_class:
    DMDana_ini_configparser0 = configparser.ConfigParser(inline_comment_prefixes="#")
    if (os.path.isfile(configfile_path)):
        default_ini=check_and_get_path(libpath+'/DMDana/do/DMDana_default.ini')
        DMDana_ini_configparser0.read([default_ini,configfile_path])
    else:
        raise Warning('%s not exist. Default setting would be used. You could run "DMDana init" to initialize it.'%configfile_path)
    return DMDana_ini_config_setting_class(**dict( ('section_'+key.replace('-','_'),val)for key,val in DMDana_ini_configparser0.items()))
    

