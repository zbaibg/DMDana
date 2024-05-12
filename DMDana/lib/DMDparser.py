import glob
import os
from typing import List

import numpy as np

from ..lib.param_in_class import param_class
from . import constant as const


def check_and_get_path(filepath: str) -> str:
    """
    Check if the given file path exists and return it if so.

    :param filepath: The path to the file to check.
    :return: The file path if it exists.
    :raises AssertionError: If the file does not exist.
    """
    assert os.path.isfile(filepath), f"{filepath} does not exist."
    return filepath


def read_text_from_file(filepath: str, marklist: List[str], locationlist: List[int], stop_at_first_find: bool,
                        dtypelist: type or List[type] = str, sep: str = None) -> List:
    """
    Read text from a file based on specified markers and locations.

    :param filepath: The path to the file to read from.
    :param marklist: A list of markers to search for in the file.
    :param locationlist: A list of locations corresponding to the markers.
    :param stop_at_first_find: Whether to stop searching after the first match for each marker.
    :param dtypelist: The data type or a list of data types to convert the results to (default: str).
    :param sep: The separator to use when splitting lines (default: None).
    :return: A list of results extracted from the file.
    :raises AssertionError: If marklist and locationlist have different lengths, or if dtypelist is not a type or list of types.

    Usage:
    To get mu and temperature from ldbd_size.dat, in which the content is like:
    9.50043414701698e-04 # T
    0.00000000000000e+00  0.00000000000000e+00  0.00000000000000e+00 # muMin, muMax mu
    just use:
    mu, temperature = read_text_from_file('ldbd_data/ldbd_size.dat', ["mu", "# T"], [2, 0])
    Remember to manually convert the string to the data type you want.
    dtypelist could either be a type or a list of types.
    """
    assert len(marklist) == len(locationlist), "marklist and locationlist should have the same length."
    assert isinstance(dtypelist, type) or isinstance(dtypelist, list), "dtypelist should be a type or a list of types."

    def loop(filepath, marklist, stop_at_first_find, locationlist):
        resultlist = [None] * len(marklist)
        with open(filepath) as f:
            for line in f:
                for i in range(len(marklist)):
                    if stop_at_first_find and None not in resultlist:
                        return resultlist
                    if marklist[i] not in line:
                        continue
                    if stop_at_first_find:
                        resultlist[i] = line.split(sep)[locationlist[i]] if resultlist[i] is None else resultlist[i]
                    else:
                        resultlist[i] = line.split(sep)[locationlist[i]]
        return resultlist

    resultlist = loop(filepath, marklist, stop_at_first_find, locationlist)
    if isinstance(dtypelist, type):
        resultlist = [dtypelist(i) if i is not None else None for i in resultlist]
    elif isinstance(dtypelist, list):
        assert len(dtypelist) == len(resultlist), "dtypelist and resultlist should have the same length."
        resultlist = [dtypelist[i](resultlist[i]) if resultlist[i] is not None else None for i in range(len(resultlist))]
    return resultlist


def glob_occupation_files(folder: str) -> List[str]:
    """
    Glob occupation files in the specified folder.

    :param folder: The folder to search for occupation files.
    :return: A list of occupation file paths.
    """
    occup_files = glob.glob(folder + '/occupations_t0.out') + sorted(glob.glob(folder + '/occupations-[0-9][0-9][0-9][0-9][0-9].out'))
    return occup_files


def get_total_step_number(folder: str) -> int:
    """
    Get the total number of steps from the jx_elec_tot.out file in the specified folder.

    :param folder: The folder containing the jx_elec_tot.out file.
    :return: The total number of steps.
    :raises AssertionError: If the jx_elec_tot.out file is not found in the folder.
    """
    linenumber = 0
    assert os.path.isfile(folder + '/jx_elec_tot.out'), f"jx_elec_tot.out not found in {folder}"
    with open(folder + '/jx_elec_tot.out') as f:
        for line in f:
            linenumber += 1
    return linenumber - 3  # not include the step at t=0


def get_current_data(folder: str, elec_or_hole: str) -> tuple:
    """
    Loads current data from specified files based on particle type (electron, hole, or total).

    :param folder: Directory path where the data files are stored.
    :param elec_or_hole: Type of particles - 'elec' for electrons, 'hole' for holes, or 'total' for combined data.
    :return: Three numpy.ndarrays corresponding to the 'x', 'y', and 'z' components of the data.
    :raises AssertionError: If an invalid particle type is specified or if the required data files are not found.
    """

    def load_data(folder: str, particle_type: str, component: str) -> np.ndarray:
        """
        Helper function to load data from a file based on folder path, particle type, and component.

        :param folder: Directory path where the data files are stored.
        :param particle_type: 'elec' or 'hole' indicating the type of particle.
        :param component: 'x', 'y', or 'z' indicating the data component.
        :return: Array of data loaded from the specified file.
        :raises AssertionError: If the required data file is not found.
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


def get_DMD_param(path: str = '.') -> param_class:
    """
    Get the parameters from the param.in file.

    :param path: The folder containing the param.in file (default: '.').
    :return: A param_class object containing the parameters from param.in.
    :raises AssertionError: If the param.in file is not correctly formatted.
    """
    DMDparam_value = dict()
    filepath = check_and_get_path(path + '/param.in')
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line == '' or line[0] == '#':
                continue
            line = line.split('#')[0]
            assert '=' in line, "param.in is not correctly formatted."
            list_for_this_line = line.split('=')
            assert len(list_for_this_line) == 2, "param.in is not correctly formatted."
            DMDparam_value[list_for_this_line[0].strip()] = list_for_this_line[1].strip()
    return param_class(**DMDparam_value)


def get_mu_temperature(DMDparam_value: dict, path: str = '.') -> tuple:
    """
    Get the chemical potential (mu) and temperature from ldbd_data/ldbd_size.dat, param.in, and out(DMD.out) files.

    :param DMDparam_value: A dictionary containing the parameters from param.in.
    :param path: The folder containing the required files (default: '.').
    :return: A tuple containing the chemical potential (mu_au) and temperature (temperature_au) in atomic units.
    :raises AssertionError: If the required files or parameters are not found.
    """
    # read ldbd_size.dat
    filepath = check_and_get_path(path + '/ldbd_data/ldbd_size.dat')
    mu_au_text, temperature_au_text = read_text_from_file(filepath, marklist=["mu", "# T"], locationlist=[2, 0], stop_at_first_find=True)
    assert mu_au_text is not None, "mu not found in ldbd_size.dat"
    assert temperature_au_text is not None, "temperature not found in ldbd_size.dat"
    mu_au = float(mu_au_text)
    temperature_au = float(temperature_au_text)
    # read parameter "mu" in param.in
    if 'mu' in DMDparam_value:
        mu_au = float(DMDparam_value['mu']) / const.Hatree_to_eV
    # read parameter "carrier_density" in param.in
    for _ in [0]:
        if 'carrier_density' not in DMDparam_value:
            break
        if DMDparam_value['carrier_density'] is None:
            break
        if float(DMDparam_value['carrier_density']) == 0:
            break
        assert os.path.isfile(path + '/out') or os.path.isfile(path + '/DMD.out'), "out or DMD.out file not found (for determining mu from non-zero carrier_density)"
        output_file_name = path + '/out' if os.path.isfile(path + '/out') else path + '/DMD.out'
        mu_au_text = read_text_from_file(output_file_name, marklist=["for given electron density"], locationlist=[5], stop_at_first_find=True)[0]
        assert mu_au_text is not None, "carrier_density is not zero, but mu could not be found in out or DMD.out."
        mu_au = float(mu_au_text)
    return mu_au, temperature_au


def get_erange(path: str = '.') -> List[float]:
    """
    Get different energy parameters from ldbd_data/ldbd_size.dat and ldbd_data/ldbd_erange_brange.dat files.

    :param path: The folder containing the ldbd_data folder (default: '.').
    :return: A list containing the energy parameters in atomic units:
             [EBot_probe_au, ETop_probe_au, EBot_dm_au, ETop_dm_au, EBot_eph_au, ETop_eph_au, EvMax_au, EcMin_au].
    :raises AssertionError: If the required files are not found or not correctly formatted.
    """
    filepath = check_and_get_path(path + '/ldbd_data/ldbd_size.dat')
    EBot_probe_au, ETop_probe_au, EBot_dm_au, ETop_dm_au, EBot_eph_au, ETop_eph_au = read_text_from_file(
        filepath, marklist=['# EBot_probe, ETop_probe, EBot_dm, ETop_dm, EBot_eph, ETop_eph'] * 6,
        locationlist=range(6), stop_at_first_find=True)
    filepath = check_and_get_path(path + '/ldbd_data/ldbd_erange_brange.dat')
    with open(filepath) as f:
        line = f.readline().split()
        assert len(line) == 2, "The first line of ldbd_erange_brange.dat is not correctly formatted."
        EvMax_au, EcMin_au = line
    return [float(i) for i in [EBot_probe_au, ETop_probe_au, EBot_dm_au, ETop_dm_au, EBot_eph_au, ETop_eph_au, EvMax_au, EcMin_au]]

