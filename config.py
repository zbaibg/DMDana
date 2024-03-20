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
'''
def read_text_from_file(filepath,marklist,locationlist):
    assert len(marklist)==len(locationlist),"marklist and locationlist should have the same length."
    resultlist=[None]*len(marklist)
    with open(filepath) as f:
        for line in f:
            for i in range(len(marklist)):
                if marklist[i] in line:
                    resultlist[i]=line.split()[locationlist[i]]
    return resultlist

def get_current_data(folder):
    jxpath=folder+"/jx_elec_tot.out"
    jypath=folder+"/jy_elec_tot.out"
    jzpath=folder+"/jz_elec_tot.out"
    jx_data=np.loadtxt(jxpath,skiprows=1)
    jy_data=np.loadtxt(jypath,skiprows=1)
    jz_data=np.loadtxt(jzpath,skiprows=1)
    return jx_data,jy_data,jz_data

def get_DMD_param(path='.'):
    DMDparam_value=dict()
    filepath = check_and_get_path(path+'/param.in')
    with open(filepath) as f:
        for line in f:
            line=line.strip()
            if line=='':
                continue
            elif line[0]=='#':
                continue
            else:
                line=line.split('#')[0]
                assert '=' in line, "param.in is not correctly setted."
                list_for_this_line=line.split('=')
                assert len(list_for_this_line)==2,"param.in is not correctly setted."
                DMDparam_value[list_for_this_line[0].strip()]=list_for_this_line[1].strip()
    return DMDparam_value

def get_mu_temperature(DMDparam_value,path='.'):
    filepath = check_and_get_path(path+'/ldbd_data/ldbd_size.dat')
    mu_au_text,temperature_au_text=read_text_from_file(filepath,marklist=["mu","# T"],locationlist=[2,0])
    assert mu_au_text != None,"mu not found in ldbd_size.dat"
    assert temperature_au_text != None, "temperature not found in ldbd_size.dat"
    mu_au=float(mu_au_text)
    temperature_au=float(temperature_au_text)
    if 'mu' in DMDparam_value:
        mu_au=float(DMDparam_value['mu'])/Hatree_to_eV
    if 'carrier_density' in DMDparam_value and float(DMDparam_value['carrier_density'])!=0:
        assert os.path.isfile('out'), "out file not found(for determine mu from non-zero carrier_density)"
        mu_au_text=read_text_from_file('out',marklist=["for given electron density"],locationlist=[5],defaultlist=[mu_au])
        mu_au=float(mu_au_text) if mu_au_text!=None else mu_au
    return mu_au,temperature_au

def get_erange(path='.'):
    filepath = check_and_get_path(path+'/ldbd_data/ldbd_size.dat')
    EBot_probe_au, ETop_probe_au, EBot_dm_au, ETop_dm_au, EBot_eph_au, ETop_eph_au=read_text_from_file(filepath,marklist=['# EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph']*6,locationlist=range(6))
    filepath= check_and_get_path(path+'/ldbd_data/ldbd_erange_brange.dat')
    with open(filepath) as f:
        line=f.readline().split()
        assert len(line)==2, "The first line of ldbd_erange_brange.dat is not correctly setted."
        EvMax_au,EcMin_au=line
    return [float(i) for i in [EBot_probe_au, ETop_probe_au, EBot_dm_au, ETop_dm_au, EBot_eph_au, ETop_eph_au ,EvMax_au, EcMin_au]]

class config_base(object): 
    def __init__(self,funcname_in,param_path='./DMDana.ini',putlog=True):
        self.libpath='/'.join(__file__.split('/')[0:-1])
        self.logfile=None
        self.config = configparser.ConfigParser(inline_comment_prefixes="#")
        if (os.path.isfile(param_path)):
            default_ini=check_and_get_path(self.libpath+'/DMDana_default.ini')
            self.config.read([default_ini,param_path])
        else:
            raise ValueError('%s not exist. Please run "DMDana init" to initialize it.'%param_path)
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
        self.Input=self.config[self.funcname]
        if putlog==True:
            self.initiallog(self.funcname)
        self.folders=[i.strip() for i in self.Input['folders'].split(',')] 
        self.folder_number=len(self.folders)
        self.DMDparam_value=get_DMD_param(self.folders[0])# Use the param.in in the first folder

    def initiallog(self,funcname):#this should be done after setting global variable "funcname" 
        repo = git.Repo(sys.path[0],search_parent_directories=True)
        sha = repo.head.object.hexsha
        logging.info("============DMDana============")
        logging.info("Git hash %s (%s)"%(sha[:7],sha))
        logging.info("Submodule: %s"%funcname)
        logging.info("Start time: %s"%datetime.datetime.now())
        logging.info("===Configuration Parameter===")
        paramdict=dict((self.config.items(funcname)))
        for i in paramdict:
            logging.info("%-35s"%i+':\t'+paramdict[i]+'')
        logging.info("===Initialization finished===")
        
class config_current(config_base):
    def __init__(self, funcname_in,param_path='./DMDana.ini',putlog=True):
        super().__init__(funcname_in,param_path,putlog)
        self.only_jtot=None
        self.jx_data=None
        self.jy_data=None
        self.jz_data=None
        self.jx_data_path=None
        self.jy_data_path=None
        self.jz_data_path=None
        self.loadcurrent_ith(0)# load the first folder by difault
        pumpPoltype=self.DMDparam_value['pumpPoltype']
        pumpA0=float(self.DMDparam_value['pumpA0'])
        pumpE=float(self.DMDparam_value['pumpE'])
        self.light_label=' '+'for light of %s Polarization, %.2e a.u Amplitude, and %.2e eV Energy'%(pumpPoltype,pumpA0,pumpE)
        
    # read data in the i_th folder provided by "folders" parameter in DMDana.ini
    def loadcurrent_ith(self,i):
        if i>=len(self.folders) or i<-len(self.folders):
            raise ValueError("i is out of range.")
        folder=self.folders[i]
        #self.config.read('DMDana.ini')
        #self.Input=self.config[self.funcname]
        self.only_jtot=self.Input.getboolean('only_jtot')
        if self.only_jtot==None:
            raise ValueError('only_jtot is not correct setted.')
        self.jx_data,self.jy_data,self.jz_data=get_current_data(folder)
        if not (len(self.jx_data)==len(self.jy_data)==len(self.jz_data)):
            raise ValueError('The line number in jx_data jy_data jz_data are not the same. Please deal with your data.' )

class config_occup(config_base):
    def __init__(self, funcname_in,param_path='./DMDana.ini',putlog=True):
        super().__init__(funcname_in,param_path,putlog)
        self.occup_timestep_for_selected_file_fs=None
        self.occup_timestep_for_selected_file_ps=None
        self.occup_t_tot=None # maximum time to plot
        self.occup_maxmium_file_number_plotted_exclude_t0=None # maximum number of files to plot exclude t0
        self.occup_selected_files=None #The filelists to be dealt with. Note: its length might be larger than 
                                # occup_maxmium_file_number_plotted_exclude_t0+1 for later possible 
                                # calculations, eg. second-order finite difference
        #self.config.read('DMDana.ini')
        #self.Input=self.config[self.funcname]
        # Read all the occupations file names at once
        if glob.glob('occupations_t0.out')==[]:
            raise ValueError("Did not found occupations_t0.out")
        self.mu_au,self.temperature_au=get_mu_temperature(self.DMDparam_value,path=self.folders[0])
        self.EBot_probe_au,self.ETop_probe_au,self.EBot_dm_au,self.ETop_dm_au,self.EBot_eph_au,self.ETop_eph_au,self.EvMax_au,self.EcMin_au=get_erange(path=self.folders[0])
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
def autoconfig(funcname):
    if funcname in ['FFT-DC-convergence-test','current-plot','FFT-spectrum-plot']:
        return config_current(funcname)
    if funcname in ['occup-time','occup-deriv']:
        return config_occup(funcname)