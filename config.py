import numpy as np
import configparser
import glob
import git
import datetime
import sys
import os
from constant import *
import logging
"""_summary_
1. Import common options from configuration files.
"""
class configclass: 
    def __init__(self,funcname_in):
        self.logfile=None
        self.config = configparser.ConfigParser(inline_comment_prefixes="#")
        if (os.path.isfile('DMDana.ini')):
            self.config.read('DMDana.ini')
        else:
            raise ValueError('DMDana.ini does not exist. Please run "DMDana.py init" to initialize it.')

        self.Input=None
        self.only_jtot=None
        self.jx_data=None
        self.jy_data=None
        self.jz_data=None
        self.light_label=None
        self.occup_timestep_for_selected_file_fs=None
        self.occup_timestep_for_selected_file_ps=None
        self.occup_t_tot=None # maximum time to plot
        self.occup_maxmium_file_number_plotted_exclude_t0=None # maximum number of files to plot exclude t0
        self.occup_selected_files=None #The filelists to be dealt with. Note: its length might be larger than 
                                # occup_maxmium_file_number_plotted_exclude_t0+1 for later possible 
                                # calculations, eg. second-order finite difference
        self.funcname=funcname_in
        self.initiallog()#this should be done after setting global variable "funcname" 
        if self.funcname in ['FFT-DC-convergence-test','current-plot','FFT-spectrum-plot']:
            self.init_current()
        if self.funcname in ['occup-time','occup-deriv']:
            self.init_occup()
        
    def init_current(self,):
        self.config.read('DMDana.ini')
        self.Input=self.config[self.funcname]
        self.only_jtot=self.Input.getboolean('only_jtot')
        if self.only_jtot==None:
            raise ValueError('only_jtot is not correct setted.')
        self.jx_data = np.loadtxt(self.Input['jx_data'],skiprows=1)
        self.jy_data = np.loadtxt(self.Input['jy_data'],skiprows=1)
        self.jz_data = np.loadtxt(self.Input['jz_data'],skiprows=1)
        if not (len(self.jx_data)==len(self.jy_data)==len(self.jz_data)):
            raise ValueError('The line number in jx_data jy_data jz_data are not the same. Please deal with your data.' )
        self.light_label=' '+self.Input['light_label']

    def init_occup(self,):
        self.config.read('DMDana.ini')
        self.Input=self.config[self.funcname]
        # Read all the occupations file names at once
        if glob.glob('occupations_t0.out')==[]:
            raise ValueError("Did not found occupations_t0.out")
        self.occup_selected_files = glob.glob('occupations_t0.out')+sorted(glob.glob('occupations-*out'))
        with open(self.occup_selected_files[1]) as f:
            firstline_this_file=f.readline()
            t1_fs=float(firstline_this_file.split()[12])/fs 
        with open(self.occup_selected_files[2]) as f:
            firstline_this_file=f.readline()
            t2_fs=float(firstline_this_file.split()[12])/fs
                     
        #occup_timestep_for_all_files=self.Input.getint('timestep_for_occupation_output') #fs
        occup_timestep_for_all_files=t2_fs-t1_fs #fs
        filelist_step=self.Input.getint('filelist_step') # select parts of the filelist
        self.occup_timestep_for_selected_file_fs=occup_timestep_for_all_files*filelist_step
        self.occup_timestep_for_selected_file_ps=self.occup_timestep_for_selected_file_fs/1000 #ps
        # Select partial files
        self.occup_selected_files=self.occup_selected_files[::filelist_step]
        self.occup_t_tot = self.Input.getint('t_max') # fs
        if self.occup_t_tot<=0:
            self.occup_maxmium_file_number_plotted_exclude_t0=len(self.occup_selected_files)-1
        else:
            self.occup_maxmium_file_number_plotted_exclude_t0=int(round(self.occup_t_tot/self.occup_timestep_for_selected_file_fs))
            if self.occup_maxmium_file_number_plotted_exclude_t0>len(self.occup_selected_files)-1:
                raise ValueError('occup_t_tot is larger than maximum time of data we have.')
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
        
    def initiallog(self,):#this should be done after setting global variable "funcname" 
        repo = git.Repo(sys.path[0],search_parent_directories=True)
        sha = repo.head.object.hexsha
        logging.info("============DMDana============")
        logging.info("Git hash %s (%s)"%(sha[:7],sha))
        logging.info("Submodule: %s"%self.funcname)
        logging.info("Start time: %s"%datetime.datetime.now())
        logging.info("===Configuration Parameter===")
        paramdict=dict((self.config.items(self.funcname)))
        for i in paramdict:
            logging.info("%-35s"%i+':\t'+paramdict[i]+'')
        logging.info("===Initialization finished===")
                            