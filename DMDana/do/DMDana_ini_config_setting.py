import configparser
import os
from typing import Any, List, Union

from pydantic import BaseModel

from ..lib.constant import libpath
from ..lib.DMDparser import check_and_get_path


class section_default_class(BaseModel):
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


class sections_current_classes(section_default_class):
    """
    A class for current section configurations.
    """
    elec_or_hole: str


class sections_occup_classes(section_default_class):
    """
    A class for occupation section configurations.
    """
    t_max: int
    filelist_step: int


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

    def model_post_init(self, __context: Any):
        """
        Post-initialization to handle the conversion of Window_type_list from string to list.
        
        :param __context: Contextual information for post-initialization.
        """
        if isinstance(self.Window_type_list, str):
            self.Window_type_list = [i.strip() for i in self.Window_type_list.split(',')]


class section_FFT_spectrum_plot_class(sections_current_classes):
    """
    A class for FFT spectrum plot section configurations.
    """
    Cutoff_list: Union[str, List[int]]
    Window_type_list: Union[str, List[str]]
    Log_y_scale: bool
    Summary_output_csv: bool
    Summary_output_xlsx: bool
    Summary_output_filename_csv: str
    Summary_output_filename_xlsx: str

    def model_post_init(self, __context: Any):
        """
        Post-initialization to handle the conversion of Window_type_list and Cutoff_list from string to list.
        
        :param __context: Contextual information for post-initialization.
        """
        if isinstance(self.Window_type_list, str):
            self.Window_type_list = [i.strip() for i in self.Window_type_list.split(',')]
        if isinstance(self.Cutoff_list, str):
            self.Cutoff_list = [int(i) for i in self.Cutoff_list.split(',')]


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


class section_occup_deriv_class(sections_occup_classes):
    """
    A class for occupation derivative section configurations.
    """
    pass


class DMDana_ini_config_setting_class(BaseModel):
    """
    A class to manage all configuration settings for DMDana.
    """
    section_DEFAULT: section_default_class
    section_current_plot: section_current_plot_class
    section_FFT_DC_convergence_test: section_FFT_DC_convergence_test_class
    section_FFT_spectrum_plot: section_FFT_spectrum_plot_class
    section_occup_time: section_occup_time_class
    section_occup_deriv: section_occup_deriv_class

    def __getitem__(self, key):
        """
        Retrieve a section configuration using its key.
        
        :param key: The key of the section to retrieve.
        :return: The section configuration object.
        """
        return getattr(self, 'section_' + key.replace('-', '_'))

    def items(self, key):
        """
        Retrieve all items of a section as a dictionary.
        
        :param key: The key of the section.
        :return: A dictionary of all items in the section.
        """
        return dict((key, str(val)) for key, val in self[key].model_dump().items())


def get_DMDana_ini_config_setting(configfile_path) -> DMDana_ini_config_setting_class:
    """
    Load and return the DMDana configuration settings from a file.
    
    :param configfile_path: The path to the configuration file.
    :return: An instance of DMDana_ini_config_setting_class with loaded settings.
    """
    DMDana_ini_configparser0 = configparser.ConfigParser(inline_comment_prefixes="#")
    if os.path.isfile(configfile_path):
        default_ini = check_and_get_path(libpath + '/DMDana/do/DMDana_default.ini')
        DMDana_ini_configparser0.read([default_ini, configfile_path])
    else:
        raise Warning('%s not exist. Default setting would be used. You could run "DMDana init" to initialize it.' % configfile_path)
    return DMDana_ini_config_setting_class(**dict(('section_' + key.replace('-', '_'), val) for key, val in DMDana_ini_configparser0.items()))


