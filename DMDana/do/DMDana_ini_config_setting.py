import configparser
import os
from typing import Any, Dict, List, Union

from pydantic import BaseModel, field_validator
from pydantic.dataclasses import dataclass

from ..lib.constant import allfuncname, libpath
from ..lib.DMDparser import check_and_get_path


@dataclass
class section_default_class:
    """
    A class for default section configurations, compatible with configparser.
    """
    only_jtot: bool
    folder: str

    def __getitem__(self, key):
        """
        Retrieve an item using its key, for compatibility with configparser.
        
        :param key: The key of the item to retrieve.
        :return: The string representation of the item.
        """
        return str(getattr(self, key))

    def __setitem__(self, key, value_str):
        """
        Set an item using its key and value, for compatibility with configparser.
        
        :param key: The key of the item to set.
        :param value_str: The string value to set for the item.
        """
        type_key = type(getattr(self, key))
        setattr(self, key, type_key(value_str))

    def getint(self, key):
        """
        Get an integer configuration value.
        
        :param key: The key of the integer configuration.
        :return: The integer value.
        """
        return int(getattr(self, key))

    def getfloat(self, key):
        """
        Get a float configuration value.
        
        :param key: The key of the float configuration.
        :return: The float value.
        """
        return float(getattr(self, key))

    def getboolean(self, key):
        """
        Get a boolean configuration value.
        
        :param key: The key of the boolean configuration.
        :return: The boolean value.
        """
        return bool(getattr(self, key))

    get = __getitem__

@dataclass
class sections_current_classes(section_default_class):
    """
    A class for current section configurations.
    """
    elec_or_hole: str

@dataclass
class sections_occup_classes(section_default_class):
    """
    A class for occupation section configurations.
    """
    t_max: int
    filelist_step: int

@dataclass
class section_current_plot_class(sections_current_classes):
    """
    A class for current plot section configurations.
    """
    current_plot_output: str
    t_min: float
    t_max: float
    smooth_on: bool
    smooth_method: str
    smooth_times: int
    smooth_windowlen: int
    plot_all: bool

@dataclass
class section_FFT_DC_convergence_test_class(sections_current_classes):
    """
    A class for FFT DC convergence test section configurations.
    """
    Cutoff_step: int
    Cutoff_min: int
    Cutoff_max: int
    Window_type_list: Union[str, List[str]]
    Database_output_csv: bool
    Database_output_xlsx: bool
    Database_output_filename_csv: str
    Database_output_filename_xlsx: str
    Figure_output_filename: str

    def __post_init__(self):
        """
        Post-initialization to handle the conversion of Window_type_list from string to list.
        """
        if isinstance(self.Window_type_list, str):
            self.Window_type_list = [i.strip() for i in self.Window_type_list.split(',')]

@dataclass(config=dict(validate_assignment=True)) #make this class's attribute be type-validated/converted when assigning
class section_FFT_spectrum_plot_class(sections_current_classes):
    """
    A class for FFT spectrum plot section configurations.
    """
    Cutoff_list: List[int]
    Window_type_list: List[str]
    Log_y_scale: bool
    Summary_output_csv: bool
    Summary_output_xlsx: bool
    Summary_output_filename_csv: str
    Summary_output_filename_xlsx: str
    
    @field_validator('Window_type_list',mode='before')
    @classmethod
    def Window_type_to_list(cls, Window_type_list: Any) -> List:
        if isinstance(Window_type_list, str):
            return [i.strip() for i in Window_type_list.split(',')]
    
    @field_validator('Cutoff_list',mode='before')
    @classmethod
    def Cutoff_list_to_list(cls, Cutoff_list: Any) -> List:
        if isinstance(Cutoff_list, str):
            return [int(i) for i in Cutoff_list.split(',')]
        if isinstance(Cutoff_list, int):
            return [Cutoff_list]
        if isinstance(Cutoff_list, float):
            return [int(Cutoff_list)]

@dataclass
class section_occup_time_class(sections_occup_classes):
    """
    A class for occupation time section configurations.
    """
    occup_time_plot_set_Erange: bool
    occup_time_plot_lowE: float
    occup_time_plot_highE: float
    plot_conduction_valence: bool
    plot_occupation_number_setlimit: bool
    plot_occupation_number_min: float
    plot_occupation_number_max: float
    output_all_figure_types: bool
    figure_style: str
    fit_Boltzmann: bool
    fit_Boltzmann_initial_guess_mu: float
    fit_Boltzmann_initial_guess_mu_auto: bool
    fit_Boltzmann_initial_guess_T: float
    fit_Boltzmann_initial_guess_T_auto: bool
    Substract_initial_occupation: bool
    showlegend: bool

@dataclass
class section_occup_deriv_class(sections_occup_classes):
    """
    A class for occupation derivative section configurations.
    """
    pass

@dataclass
class DMDana_ini_config_setting_class:
    """
    A class to manage all configuration settings for DMDana.
    """
    section_DEFAULT: section_default_class
    section_current_plot: section_current_plot_class
    section_FFT_DC_convergence_test: section_FFT_DC_convergence_test_class
    section_FFT_spectrum_plot: section_FFT_spectrum_plot_class
    section_occup_time: section_occup_time_class
    section_occup_deriv: section_occup_deriv_class
    
    '''def __getitem__(self, key: str):
        """
        Retrieve a section configuration using its key.
        
        :param key: The key of the section to retrieve.
        :return: The section configuration object.
        """
        return getattr(self, 'section_' + key.replace('-', '_'))'''

    '''def items(self, key):
        """
        Retrieve all items of a section as a dictionary.
        
        :param key: The key of the section.
        :return: A dictionary of all items in the section.
        """
        return dict((key, str(val)) for key, val in self[key].__dict__.items())'''

    def get_section_from_funcname(self, funcname: str):
        assert funcname in allfuncname, f"funcname is not valid, it should be in {allfuncname}"
        return getattr(self,'section_'+funcname)

def get_DMDana_ini_config_setting(configfile_path: str) -> DMDana_ini_config_setting_class:
    """
    Load and return the DMDana configuration settings from a file.
    
    :param configfile_path: The path to the configuration file.
    :return: An instance of DMDana_ini_config_setting_class with loaded settings.
    """
    DMDana_ini_configparser0 = configparser.ConfigParser(inline_comment_prefixes="#")
    DMDana_ini_configparser0.optionxform = str
    if os.path.isfile(configfile_path):
        default_ini = check_and_get_path(libpath + '/DMDana/do/DMDana_default.ini')
        DMDana_ini_configparser0.read([default_ini, configfile_path])
    else:
        raise Warning('%s not exist. Default setting would be used. You could run "DMDana init" to initialize it.' % configfile_path)
    
    def class_convert(config_parser: configparser.ConfigParser):
        """
        Generate an instance of this class from a configparser instance.
        :param config_parser: configparser.ConfigParser instance, containing all the configurations.
        :return: DMDana_ini_config_setting_class instance.
        """
        dict_all = {}
        Field_Class_dict = {sec_var_name: section_name_class for sec_var_name, section_name_class in DMDana_ini_config_setting_class.__annotations__.items()}

        for section in list(config_parser.sections())+['DEFAULT']:
            section_var_name = 'section_' + section.replace('-', '_')
            if section_var_name in Field_Class_dict:
                section_class = Field_Class_dict[section_var_name]
                dict_all[section_var_name] = section_class(**dict(config_parser.items(section)))
            else:
                raise ValueError(f"No field found for section {section_var_name} in DMDana_ini_config_setting_class")
        return DMDana_ini_config_setting_class(**dict_all)
    
    return class_convert(DMDana_ini_configparser0)

def change_folder_for_DMDana_ini_config_setting(DMDana_ini_config_setting:DMDana_ini_config_setting_class,change_DMD_folder_to:str):
    """
    change DMD folder in the DMDana configuration settings.
    
    :param DMDana_ini_config_setting: DMDana_ini_config_setting_class instance.
    :param change_DMD_folder_to: change the DMD folder in the DMDana.ini to this folder.
    :return: An instance of DMDana_ini_config_setting_class with loaded settings.
    """
    for section_key,section_data in DMDana_ini_config_setting.__dict__.items():
        section_data: Union[section_default_class,Any]
        section_data.folder=change_DMD_folder_to
    return DMDana_ini_config_setting
