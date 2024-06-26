from pydantic import BaseModel

class param_class(BaseModel):
    DEBUG:float=None
    restart:float=None
    code:str=None
    material_model:str=None
    compute_tau_only:float=None
    alg_scatt_enable:float=None
    alg_eph_enable:float=None
    modeStart:float=None
    modeEnd:float=None
    scale_scatt:float=None
    scale_eph:float=None
    scale_ei:float=None
    scale_ee:float=None
    alg_only_eimp:float=None
    alg_only_ee:float=None
    alg_only_intravalley:float=None
    alg_only_intervalley:float=None
    alg_eph_sepr_eh:float=None
    alg_eph_need_elec:float=None
    alg_eph_need_hole:float=None
    alg_semiclassical:float=None
    alg_summode:float=None
    alg_ddmdteq:float=None
    alg_expt:float=None
    alg_expt_elight:float=None
    alg_scatt:str=None
    alg_picture:str=None
    alg_linearize:float=None
    alg_linearize_dPee:float=None
    need_imsig:float=None
    alg_modelH0hasBS:float=None
    alg_read_Bso:float=None
    alg_Pin_is_sparse:float=None
    alg_sparseP:float=None
    alg_thr_sparseP:float=None
    alg_phenom_relax:float=None
    tau_phenom:float=None
    bStart_tau:float=None
    bEnd_tau:float=None
    alg_phenom_recomb:float=None
    tau_phenom_recomb:float=None
    degauss:float=None
    ndegauss:float=None
    temperature:float=None
    nk1:float=None
    nk2:float=None
    nk3:float=None
    ewind:float=None
    lattvec1:float=None
    lattvec2:float=None
    lattvec3:float=None
    dim:float=None
    thickness:float=None
    scissor:float=None
    mu:float=None
    carrier_density:float=None
    carrier_density_means_excess_density:float=None
    Bx:float=None
    By:float=None
    Bz:float=None
    gfac_normal_dist:float=None
    gfac_k_resolved:float=None
    gfac_mean:float=None
    gfac_sigma:float=None
    gfac_cap:float=None
    scale_Ez:float=None
    scrMode:str=None
    scrFormula:str=None
    update_screening:float=None
    dynamic_screening:str=None
    ppamodel:str=None
    eppa_screening:float=None
    meff_screening:float=None
    omegamax_screening:float=None
    nomega_screening:float=None
    dynamic_screening_ee_two_freqs:float=None
    fderavitive_technique_static_screening:float=None
    smearing_screening:float=None
    epsilon_background:float=None
    impurity_density:float=None
    impMode:str=None
    partial_ionized:float=None
    Z_impurity:float=None
    g_impurity:float=None
    E_impurity:float=None
    degauss_eimp:float=None
    detailBalance:float=None
    freq_update_eimp_model:float=None
    eeMode:str=None
    ee_antisymmetry:float=None
    degauss_ee:float=None
    freq_update_ee_model:float=None
    laserMode:str=None
    pumpMode:str=None
    laserAlg:str=None
    pumpA0:float=None
    laserA:float=None
    pumpE:float=None
    laserE:float=None
    pumpTau:float=None
    pump_tcenter:float=None
    pumpPoltype:str=None
    laserPoltype:str=None
    ExEyPolAngle:float=None
    probePoltype:str=None
    probeEmin:float=None
    probeEmax:float=None
    probeDE:float=None
    probeTau:float=None
    Bxpert:float=None
    Bypert:float=None
    Bzpert:float=None
    needL:float=None
    t0:float=None
    tend:float=None
    tstep:float=None
    tstep_pump:float=None
    tstep_laser:float=None
    print_tot_band:float=None
    alg_set_scv_zero:float=None
    freq_measure:float=None
    freq_measure_ene:float=None
    freq_compute_tau:float=None
    de_measure:float=None
    degauss_measure:float=None
    occup_write_interval:float=None
    valley:float=None
    file_forbid_vtrans:str=None
    type_q_ana:str=None
    rotate_spin_axes:float=None
    sdir_z:float=None
    sdir_x:float=None
    alg_ode_method:str=None
    ode_hstart:float=None
    ode_hmin:float=None
    ode_hmax:float=None
    ode_hmax_laser:float=None
    ode_epsabs:float=None
    print_along_kpath:float=None
    kpath_start:float=None
    kpath_end:float=None
    alg_use_dmDP_taufm_as_init:float=None
    alg_DP_beyond_carrierlifetime:float=None
    alg_mix_tauneq:float=None
    alg_positive_tauneq:float=None
    alg_use_dmDP_in_evolution:float=None
    degthr:float=None
    band_skipped:float=None
    def __getitem__(self, key):
        """
        Retrieve the string representation of the attribute specified by `key`.

        :param key: The attribute name whose value is to be retrieved.
        :type key: str
        :return: The string representation of the attribute value.
        :rtype: str
        """
        return str(getattr(self, key))

    def __setitem__(self, key, value_str):
        """
        Set the attribute specified by `key` to a new value, converting `value_str` to the appropriate type.

        :param key: The attribute name whose value is to be set.
        :type key: str
        :param value_str: The new value for the attribute, provided as a string.
        :type value_str: str
        """
        type_key = type(getattr(self, key))
        setattr(self, key, type_key(value_str))
