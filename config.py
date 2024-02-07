import numpy as np
import configparser
import glob
import git
import datetime
import sys
from constant import *
"""_summary_
1. Import common options from configuration files.
2. Include scientific constants.
"""

logfile=None
config = configparser.ConfigParser(inline_comment_prefixes="#")
config.read('DMDana.ini')

Input=None
only_jtot=None
jx_data=None
jy_data=None
jz_data=None
light_label=None
occup_timestep_for_selected_file_fs=None
occup_timestep_for_selected_file_ps=None
occup_t_tot=None # maximum time to plot
occup_maxmium_file_number_plotted_exclude_t0=None # maximum number of files to plot exclude t0
occup_selected_files=None #The filelists to be dealt with. Note: its length might be larger than 
                        # occup_maxmium_file_number_plotted_exclude_t0+1 for later possible 
                        # calculations, eg. second-order finite difference
funcname=None


def init(funcname_in):
    global funcname,logfile
    funcname=funcname_in
    initiallog()#this should be done after setting global variable "funcname" 
    if funcname in ['FFT-DC-convergence-test','current-plot','FFT-spectrum-plot']:
        init_current()
    if funcname in ['occup-time','occup-deriv']:
        init_occup()
    
def end():
    endlog()

def init_current():
    global config,Input,only_jtot,jx_data,jy_data,jz_data,light_label
    config.read('DMDana.ini')
    Input=config[funcname]
    only_jtot=Input.getboolean('only_jtot')
    if only_jtot==None:
        raise ValueError('only_jtot is not correct setted.')
    jx_data = np.loadtxt(Input['jx_data'],skiprows=1)
    jy_data = np.loadtxt(Input['jy_data'],skiprows=1)
    jz_data = np.loadtxt(Input['jz_data'],skiprows=1)
    if jx_data.shape[0]!= jy_data.shape[0] or jx_data.shape[0]!= jz_data.shape[0] or jy_data.shape[0]!= jz_data.shape[0]:
        raise ValueError('The line number in jx_data jy_data jz_data are not the same. Please deal with your data.' )
    light_label=' '+Input['light_label']

def init_occup():
    global config,Input,occup_timestep_for_selected_file_fs,occup_timestep_for_selected_file_ps
    global occup_t_tot,occup_maxmium_file_number_plotted_exclude_t0,occup_selected_files
    config.read('DMDana.ini')
    Input=config[funcname]
    occup_timestep_for_all_files=Input.getint('timestep_for_occupation_output') #fs
    filelist_step=Input.getint('filelist_step') # select parts of the filelist
    occup_timestep_for_selected_file_fs=occup_timestep_for_all_files*filelist_step
    occup_timestep_for_selected_file_ps=occup_timestep_for_selected_file_fs/1000 #ps
    # Read all the occupations file names at once
    if glob.glob('occupations_t0.out')==[]:
        raise ValueError("Did not found occupations_t0.out")
    occup_selected_files = glob.glob('occupations_t0.out')+sorted(glob.glob('occupations-*out'))
    # Select partial files
    occup_selected_files=occup_selected_files[::filelist_step]
    occup_t_tot = Input.getint('t_max') # fs
    if occup_t_tot<=0:
        occup_maxmium_file_number_plotted_exclude_t0=len(occup_selected_files)-1
    else:
        occup_maxmium_file_number_plotted_exclude_t0=int(round(occup_t_tot/occup_timestep_for_selected_file_fs))
        if occup_maxmium_file_number_plotted_exclude_t0>len(occup_selected_files)-1:
            raise ValueError('occup_t_tot is larger than maximum time of data we have.')
        occup_selected_files=occup_selected_files[:occup_maxmium_file_number_plotted_exclude_t0+2]
        # keep the first few items of the filelists in need to avoid needless calculation cost.
        # If len(occup_selected_files)-1 > occup_maxmium_file_number_plotted_exclude_t0,
        # I Add one more points larger than points needed to be plotted for later possible second-order difference 
        # calculation. However if occup_maxmium_file_number_plotted_exclude_t0 happens to be len(occup_selected_files)-1, 
        # the number of points considered in later possible second-order difference calculation is still the number of points 
        # to be plotted
        # if in the future, more file number (more than occup_maxmium_file_number_plotted_exclude_t0+2) 
        # is needed to do calculation. This should be modified.
    occup_t_tot=occup_maxmium_file_number_plotted_exclude_t0*occup_timestep_for_selected_file_fs#fs
    
def initiallog():#this should be done after setting global variable "funcname" 
    global funcname,logfile
    logfile=open('DMDana_'+funcname+'.log','w')
    repo = git.Repo(sys.path[0],search_parent_directories=True)
    sha = repo.head.object.hexsha
    logfile.write("============DMDana============\n")
    logfile.write("Git hash %s (%s)\n"%(sha[:7],sha))
    logfile.write("Submodule: %s\n"%funcname)
    logfile.write("Start time: %s\n"%datetime.datetime.now())
    logfile.write("===Configuration Parameter===\n")
    paramdict=dict((config.items(funcname)))
    for i in paramdict:
        logfile.write("%-35s"%i+':\t'+paramdict[i]+'\n')
    logfile.write("===Initialization finished===\n")
                        
def endlog():
    global funcname,logfile
    logfile.write("End time: %s\n"%datetime.datetime.now())
    logfile.close()