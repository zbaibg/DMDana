import datetime
import logging
import os
from typing import Any, List, Optional, Union

import git
import numpy as np
import pkg_resources
from pydantic.dataclasses import dataclass

from ..lib import constant as const
from ..lib.constant import libpath
from ..lib.DMDparser import (get_current_data, get_DMD_param, get_erange,
                             get_mu_temperature, glob_occupation_files)
from ..lib.param_in_class import param_class
from .DMDana_ini_config_setting import (DMDana_ini_config_setting_class,
                                        get_DMDana_ini_config_setting,
                                        section_current_plot_class,
                                        section_default_class,
                                        section_FFT_DC_convergence_test_class,
                                        section_FFT_spectrum_plot_class,
                                        section_occup_deriv_class,
                                        section_occup_time_class)

# import these to here for compatibility with old code of other files which need these

@dataclass(config=dict(arbitrary_types_allowed=True))
class config_base():

    DMDana_ini_config_setting: DMDana_ini_config_setting_class
    funcname: str = None
    show_init_log: bool = True
    log_initialized: bool = False
    DMDfolder: str= None
    DMDparam_value: param_class = None
    logger: Optional[logging.Logger] = None
    configsetting: Union[section_FFT_DC_convergence_test_class,section_FFT_spectrum_plot_class,section_current_plot_class,section_occup_deriv_class,section_occup_time_class,section_default_class] = None

    def __post_init__(self):
        assert self.funcname is not None, 'funcname is not set.'
        if self.DMDfolder is None:
            self.DMDfolder=self.DMDana_ini_config_setting.section_DEFAULT.folder
        if self.configsetting is None:
            self.configsetting = self.DMDana_ini_config_setting.get_section_from_funcname(self.funcname)
        if self.show_init_log and not self.log_initialized:
            self.initial_log(self.funcname)
            self.log_initialized = True
        if self.DMDparam_value is None:
            self.DMDparam_value = get_DMD_param(self.DMDfolder)  # Use the param.in in the first folder

    def initial_log(self, funcname):
        DMDana_version=pkg_resources.get_distribution('DMDana').version
        self.log('info',"============DMDana============")
        self.log('info',"DMDversion(with Git hash): %s" % DMDana_version)
        self.log('info',"Submodule: %s" % funcname)
        self.log('info',"Start time: %s" % datetime.datetime.now())
        self.log('info',"===Configuration Parameter===")
        paramdict = self.configsetting.__dict__
        for i in paramdict:
            self.log('info',"%-35s" % i + ':\t' + str(paramdict[i]) + '')
        self.log('info',"===Initialization finished===")
    def log(self,levelstr: str,msg: Any):
        '''
        log msg with levelstr
        
        param: levelstr : str , 'info' , 'warning' , 'error'
        param: msg : Any
        '''
        if msg is not str:
            msg=str(msg)
        if levelstr=='debug':
            Level=logging.DEBUG
        elif levelstr=='info':
            Level=logging.INFO
        elif levelstr=='warning':
            Level=logging.WARNING
        elif levelstr=='error':
            Level=logging.ERROR
        else:
            raise ValueError(f'levelstr {levelstr} is not valid, it should be either "debug", "info", "warning", or "error"')
        if self.logger is None:
            logging.log(Level, msg)
        else:
            self.logger.log(Level, msg)
@dataclass
class config_current(config_base):
    elec_or_hole_this: str = None
    bandlabel: str = None
    light_label: str = None
    jx_data: list = None
    jy_data: list = None
    jz_data: list = None

    def __post_init__(self):
        super().__post_init__()
        self.configsetting : Union[section_current_plot_class, section_FFT_spectrum_plot_class, section_FFT_DC_convergence_test_class ]
        if self.elec_or_hole_this is None:
            self.elec_or_hole_this = self.configsetting.elec_or_hole
        if self.jx_data is None or self.jy_data is None or self.jz_data is None:
            self.jx_data, self.jy_data, self.jz_data = get_current_data(self.DMDfolder, self.elec_or_hole_this)
            assert len(self.jx_data) == len(self.jy_data) == len(self.jz_data), 'The line number in jx_data, jy_data, jz_data are not the same. Please deal with your data.'
        if self.bandlabel is None:
            self.bandlabel = {'elec': 'conduction bands', 'hole': 'valence bands', 'total': 'all bands'}[self.elec_or_hole_this]
        if self.light_label is None:
            self.light_label = f' for light of {self.DMDparam_value.pumpPoltype} Polarization, {self.DMDparam_value.pumpA0:.2e} a.u Amplitude, and {self.DMDparam_value.pumpE:.2e} eV Energy for {self.bandlabel}'
        
@dataclass
class config_occup(config_base):
    mu_au: float = None
    temperature_au: float = None
    
    EBot_probe_au: float = None
    ETop_probe_au: float = None
    EBot_dm_au: float = None
    ETop_dm_au: float = None
    EBot_eph_au: float = None
    ETop_eph_au: float = None
    EvMax_au: float = None
    EcMin_au: float = None
    
    occup_Emin_au: float = None
    occup_Emax_au: float = None
    occup_timestep_for_all_files: float = None
    filelist_step: int = None
    occup_timestep_for_selected_file_fs: float = None
    occup_timestep_for_selected_file_ps: float = None
    occup_t_tot: float = None  # maximum time to plot
    occup_maxmium_file_number_plotted_exclude_t0: int = None  # maximum number of files to plot exclude t0
    occup_selected_files: List[str] = None  # The filelists to be dealt with.
    def __post_init__(self):
        super().__post_init__()
        self.configsetting: Union[section_occup_deriv_class, section_occup_time_class]
        if self.mu_au is None or self.temperature_au is None:
            self.mu_au, self.temperature_au = get_mu_temperature(self.DMDparam_value, path=self.DMDfolder)
        if self.EBot_probe_au is None or self.ETop_probe_au is None or self.EBot_dm_au is None or self.ETop_dm_au is None or self.EBot_eph_au is None or self.ETop_eph_au is None or self.EvMax_au is None or self.EcMin_au is None:
            self.EBot_probe_au, self.ETop_probe_au, self.EBot_dm_au, self.ETop_dm_au, self.EBot_eph_au, self.ETop_eph_au, self.EvMax_au, self.EcMin_au = get_erange(path=self.DMDfolder)
        if self.occup_selected_files is None or self.occup_timestep_for_all_files is None or self.filelist_step is None or self.occup_timestep_for_selected_file_fs is None or self.occup_timestep_for_selected_file_ps is None or self.occup_t_tot is None or self.occup_maxmium_file_number_plotted_exclude_t0 is None or self.occup_Emin_au is None or self.occup_Emax_au is None:
            self.occup_selected_files = glob_occupation_files(self.DMDfolder)
            assert len(self.occup_selected_files) >= 3, 'The number of occupation files is less than 3. Please check your data.'
            with open(self.occup_selected_files[1]) as f:
                firstline_this_file = f.readline()
                t1_fs = float(firstline_this_file.split()[12]) / const.fs
            with open(self.occup_selected_files[2]) as f:
                firstline_this_file = f.readline()
                t2_fs = float(firstline_this_file.split()[12]) / const.fs
            data_first = np.loadtxt(self.occup_selected_files[0])
            try:
                self.occup_Emin_au = np.min(data_first[:, 0])
                self.occup_Emax_au = np.max(data_first[:, 0])
            except Exception as e:
                logging.error('%s file is in wrong format' % self.occup_selected_files[0])
                raise e
            self.occup_timestep_for_all_files = t2_fs - t1_fs  # fs
            self.filelist_step = self.configsetting.filelist_step  # select parts of the filelist
            self.occup_timestep_for_selected_file_fs = self.occup_timestep_for_all_files * self.filelist_step
            self.occup_timestep_for_selected_file_ps = self.occup_timestep_for_selected_file_fs / 1000  # ps
            # Select partial files
            self.occup_selected_files = self.occup_selected_files[::self.filelist_step]
            self.occup_t_tot = self.configsetting.t_max  # fs
            if self.occup_t_tot <= 0:
                self.occup_maxmium_file_number_plotted_exclude_t0 = len(self.occup_selected_files) - 1
            else:
                self.occup_maxmium_file_number_plotted_exclude_t0 = int(round(self.occup_t_tot / self.occup_timestep_for_selected_file_fs))
                assert self.occup_maxmium_file_number_plotted_exclude_t0 <= len(self.occup_selected_files) - 1, 'occup_t_tot is larger than maximum time of data we have.'
                self.occup_selected_files = self.occup_selected_files[:self.occup_maxmium_file_number_plotted_exclude_t0 + 2]
            self.occup_t_tot = self.occup_maxmium_file_number_plotted_exclude_t0 * self.occup_timestep_for_selected_file_fs  # fs
