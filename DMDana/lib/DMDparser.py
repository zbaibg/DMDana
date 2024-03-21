import numpy as np
import os
from . import constant as const
import glob
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
def read_text_from_file(filepath,marklist,locationlist,stop_at_first_find):
    assert len(marklist)==len(locationlist),"marklist and locationlist should have the same length."
    resultlist=[None]*len(marklist)
    with open(filepath) as f:
        for line in f:
            for i in range(len(marklist)):
                if marklist[i] not in line:
                    continue
                if stop_at_first_find and None not in resultlist:
                    return resultlist
                if stop_at_first_find:
                    resultlist[i]=line.split()[locationlist[i]] if resultlist[i]==None else resultlist[i]
                else:
                    resultlist[i]=line.split()[locationlist[i]]
    return resultlist

def glob_occupation_files(folder):
    assert glob.glob(folder+'/occupations_t0.out')!=[], "Did not found occupations_t0.out at folder %s"%folder
    occup_files = glob.glob(folder+'/occupations_t0.out')+sorted(glob.glob(folder+'/occupations-*out'))
    return occup_files

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
        assert os.path.isfile('out') or os.path.isfile('DMD.out'), "out or DMD.out file not found(for determine mu from non-zero carrier_density)"
        output_file_name='out' if os.path.isfile('out') else 'DMD.out'
        mu_au_text=read_text_from_file(output_file_name,marklist=["for given electron density"],locationlist=[5],defaultlist=[mu_au],stop_at_first_find=True)
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
