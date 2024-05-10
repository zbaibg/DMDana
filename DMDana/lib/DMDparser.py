import glob
import os
from typing import List
import numpy as np
from . import constant as const

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
def read_text_from_file(filepath,marklist,locationlist,stop_at_first_find,dtypelist=str,sep=None) -> List:
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
                        resultlist[i]=line.split(sep)[locationlist[i]] if resultlist[i]==None else resultlist[i]
                    else:
                        resultlist[i]=line.split(sep)[locationlist[i]]
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

def get_current_data(folder, elec_or_hole):
    """
    Loads current data from specified files based on particle type (electron, hole, or total).

    Args:
    folder (str): Directory path where the data files are stored.
    elec_or_hole (str): Type of particles - 'elec' for electrons, 'hole' for holes, or 'total' for combined data.

    Returns:
    tuple: Three numpy.ndarrays corresponding to the 'x', 'y', and 'z' components of the data.
    """

    def load_data(folder, particle_type, component):
        """
        Helper function to load data from a file based on folder path, particle type, and component.

        Args:
        folder (str): Directory path where the data files are stored.
        particle_type (str): 'elec' or 'hole' indicating the type of particle.
        component (str): 'x', 'y', or 'z' indicating the data component.

        Returns:
        numpy.ndarray: Array of data loaded from the specified file.
        """
        path = f"{folder}/j{component}_{particle_type}_tot.out"
        assert os.path.isfile(path), f"File not found: {path}"
        return np.loadtxt(path, skiprows=1)

    # Ensure the input type is one of the allowed options
    assert elec_or_hole in ['elec', 'hole', 'total'], "Invalid type specified"

    if elec_or_hole == 'total':
        # Load and sum data for both electrons and holes when 'total' is requested
        data_elec = {comp: load_data(folder, 'elec', comp) for comp in ['x', 'y', 'z']}
        data_hole = {comp: load_data(folder, 'hole', comp) for comp in ['x', 'y', 'z']}

        data = {}
        for comp in ['x', 'y', 'z']:
            # Combine electron and hole data by summing the second column of arrays
            data_comp = data_elec[comp].copy()
            data_comp[:, 1:] += data_hole[comp][:, 1:]
            data[comp] = data_comp
    else:
        # Load data for a single particle type (electron or hole)
        data = {comp: load_data(folder, elec_or_hole, comp) for comp in ['x', 'y', 'z']}

    return data['x'], data['y'], data['z']


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
        if DMDparam_value['carrier_density'] ==None: 
            break
        if float(DMDparam_value['carrier_density'])==0:
            break
        assert os.path.isfile(path+'/out') or os.path.isfile(path+'/DMD.out'), "out or DMD.out file not found(for determine mu from non-zero carrier_density)"
        output_file_name=path+'/out' if os.path.isfile(path+'/out') else path+'/DMD.out'
        mu_au_text=read_text_from_file(output_file_name,marklist=["for given electron density"],locationlist=[5],stop_at_first_find=True)[0]
        assert mu_au_text != None, "carrier_density is not zero, but mu could not be found in out or DMD.out."
        mu_au=float(mu_au_text) 
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

