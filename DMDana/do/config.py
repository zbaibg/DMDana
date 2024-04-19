import configparser
import glob
import git
import datetime
import os
from ..lib import constant as const
import logging
from ..lib.DMDparser import *
from typing import Union
"""_summary_
1. Import common options from configuration files.
"""
libpath='/'.join(__file__.split('/')[0:-3])  # The path where DMDana is installed (not including DMDana)
allfuncname=['FFT_DC_convergence_test','current_plot','FFT_spectrum_plot','occup_time','occup_deriv']
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
        assert self.only_jtot!=None, 'only_jtot is not correct setted.'
        self.jx_data,self.jy_data,self.jz_data=get_current_data(self.folder)
        assert len(self.jx_data)==len(self.jy_data)==len(self.jz_data), 'The line number in jx_data jy_data jz_data are not the same. Please deal with your data.'
        pumpPoltype=self.DMDparam_value['pumpPoltype']
        pumpA0=float(self.DMDparam_value['pumpA0'])
        pumpE=float(self.DMDparam_value['pumpE'])
        self.light_label=' '+'for light of %s Polarization, %.2e a.u Amplitude, and %.2e eV Energy'%(pumpPoltype,pumpA0,pumpE)
        
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

def workflow(funcname,param_path='./DMDana.ini'):
    '''
    DMDana.ini ----------->  [ config(folder1),config(folder2)....] (results of each folder)
    
    This tree is read by different analysis modules to do the analysis.

    '''
    DMDana_ini=DMDana_ini_Class(param_path)
    assert funcname in allfuncname, 'funcname is not correct.'
    if funcname == "FFT_DC_convergence_test":
        from . import FFT_DC_convergence_test
        FFT_DC_convergence_test.do(DMDana_ini)
    elif funcname == "FFT_spectrum_plot":
        from . import FFT_spectrum_plot
        FFT_spectrum_plot.do(DMDana_ini)
    elif funcname == "current_plot":
        from . import current_plot
        current_plot.do(DMDana_ini)
    elif funcname == "occup_deriv":
        from . import occup_deriv
        occup_deriv.do(DMDana_ini)
    elif funcname == "occup_time":
        from . import occup_time
        occup_time.do(DMDana_ini)
