from parallel_folder_analysis import FolderAnalysis,parallel_folder_analysis
from DMDana.lib.constant import *
from DMDana.lib.DMDparser import read_text_from_file
from DMDana.do.occup_time import fermi
import DMDana.lib.DMDparser as DMDparser
from dataclasses import dataclass, field

def parse_param_in(fa:FolderAnalysis,func_logger):
    fa.df.loc[fa.index,'Light_pumpE']=fa.DMD_instance.param.pumpE
    fa.df.loc[fa.index,'Light_pumpPoltype']=fa.DMD_instance.param.pumpPoltype
    fa.df.loc[fa.index,'Light_pumpA0']=fa.DMD_instance.param.pumpA0
    fa.df.loc[fa.index,'Recomb_tau']=fa.DMD_instance.param.tau_phenom_recomb if fa.DMD_instance.param.alg_phenom_recomb else None
    fa.df.loc[fa.index,'impurity_density']=fa.DMD_instance.param.impurity_density

def get_Scat_State(fa:FolderAnalysis,func_logger):
    Scat_on=True if fa.DMD_instance.param.alg_scatt_enable or fa.DMD_instance.param.alg_scatt_enable==None else False
    
    RTA_used=fa.DMD_instance.param.alg_phenom_relax and Scat_on
    Fully_use_RTA=RTA_used and (fa.DMD_instance.param.bEnd_tau-fa.DMD_instance.param.bStart_tau)==(fa.DMD_instance.lindblad_init.bTop_dm-fa.DMD_instance.lindblad_init.bBot_dm)

    Use_explicit_Scat= (not Fully_use_RTA) and Scat_on 

    fa.df.loc[fa.index,'Scatt_enabled']='Off' if not Scat_on else 'On'
            
    fa.df.loc[fa.index,'RTA_tau_ps']=fa.DMD_instance.param.tau_phenom if RTA_used else None
    fa.df.loc[fa.index,'bStart_tau']=fa.DMD_instance.param.bStart_tau if RTA_used else None
    fa.df.loc[fa.index,'bEnd_tau']=fa.DMD_instance.param.bEnd_tau if RTA_used else None
    
    ePhDelta=read_text_from_file(fa.DMD_instance.lindblad_init.lindblad_folder+'/lindbladInit.out',['ePhDelta'],[2],False,float)[0]*Hatree_to_eV
    fa.df.loc[fa.index,'Eph-ePhDelta']=ePhDelta if Use_explicit_Scat else None
    fa.df.loc[fa.index,'EE-epsilon']=fa.DMD_instance.param.epsilon_background if fa.DMD_instance.param.eeMode and Use_explicit_Scat else None
    

def read_lindblad_out(fa:FolderAnalysis,func_logger):
    NkMult=read_text_from_file(fa.DMD_instance.lindblad_init.lindblad_folder+'/lindbladInit.out',marklist=['NkMult =']*3,locationlist=[3,4,5],stop_at_first_find=True,dtypelist=int)
    fa.df.loc[fa.index,'NkMult']='%s'%(NkMult)
    fa.df.loc[fa.index,'Kpoint-num']=fa.DMD_instance.lindblad_init.k_number
    fa.df.loc[fa.index,'DM_upper_bound']=fa.DMD_instance.lindblad_init.DM_Upper_E_eV
    fa.df.loc[fa.index,'DM_lower_bound']=fa.DMD_instance.lindblad_init.DM_Lower_E_eV

def get_T_and_mu(fa:FolderAnalysis,func_logger):
    fa.DMD_instance.get_mu_eV_and_T_K()
    fa.df.loc[fa.index,'mu_eV']=fa.DMD_instance.mu_eV
    fa.df.loc[fa.index,'T_K']=fa.DMD_instance.temperature_K
    func_logger.info('mu_eV=%.3f, T_K=%.1f'%(fa.DMD_instance.mu_eV,fa.DMD_instance.temperature_K))
    

def get_step_and_time(fa:FolderAnalysis,func_logger):
    fa.DMD_instance.get_total_step_num_and_total_time_fs()
    fa.df.loc[fa.index,'total_step_num']=fa.DMD_instance.total_step_num
    fa.df.loc[fa.index,'total_time_fs']=fa.DMD_instance.total_time_fs
    

def occup_time(fa:FolderAnalysis,func_logger):
    fa.DMD_instance.analyze.configsetting.section_occup_time.t_max=-1
    config_result=fa.DMD_instance.analyze.config_result.occup_time
    occup_timestep_for_all_files=config_result.occup_timestep_for_all_files
    filelist_step=int(np.round(500/occup_timestep_for_all_files))
    fa.DMD_instance.analyze.configsetting.section_occup_time.filelist_step=filelist_step
    occup_t_tot=config_result.occup_t_tot
    t_max_for_occup_time=-1 if occup_t_tot<2502 else 2502
    fa.DMD_instance.analyze.configsetting.section_occup_time.t_max=t_max_for_occup_time
    fa.DMD_instance.analyze.configsetting.section_occup_time.showlegend=True
    fa.DMD_instance.analyze.occup_time()
    

def occup_time_short_range_for_better_fit(fa:FolderAnalysis,func_logger):
    occup_time_config_tmp=fa.DMD_instance.analyze.config_result.occup_time
    occup_Emax_eV=occup_time_config_tmp.occup_Emax_au/eV
    EcMin_eV=fa.DMD_instance.lindblad_init.energy.EcMin_eV
    fa.DMD_instance.analyze.configsetting.section_occup_time.plot_conduction_valence=False
    fa.DMD_instance.analyze.configsetting.section_occup_time.occup_time_plot_set_Erange=True
    fa.DMD_instance.analyze.configsetting.section_occup_time.occup_time_plot_lowE=(occup_Emax_eV+EcMin_eV)/2
    fa.DMD_instance.analyze.configsetting.section_occup_time.occup_time_plot_highE=occup_Emax_eV
    fa.DMD_instance.analyze.occup_time()
    

def occup_deriv(fa:FolderAnalysis,func_logger):
    fa.DMD_instance.analyze.occup_deriv()
    

def current_plot(fa:FolderAnalysis,func_logger):
    fa.DMD_instance.analyze.current_plot()
    

def FFT_spectrum_plot_log(fa:FolderAnalysis,func_logger):
    fa.DMD_instance.get_total_step_num_and_total_time_fs()
    assert fa.DMD_instance.total_step_num!=None,'total_step_num is not set, please run get_total_step_num_and_total_time_fs() first'
    fa.DMD_instance.analyze.configsetting.section_FFT_spectrum_plot.Cutoff_list=max(fa.DMD_instance.total_step_num-1000,1)
    fa.DMD_instance.analyze.configsetting.section_FFT_spectrum_plot.Log_y_scale=True
    fa.DMD_instance.analyze.FFT_spectrum_plot()    
    
def FFT_spectrum_plot_linear(fa:FolderAnalysis,func_logger):
    fa.DMD_instance.get_total_step_num_and_total_time_fs()
    assert fa.DMD_instance.total_step_num!=None,'total_step_num is not set, please run get_total_step_num_and_total_time_fs() first'
    fa.DMD_instance.analyze.configsetting.section_FFT_spectrum_plot.Cutoff_list=max(fa.DMD_instance.total_step_num-1000,1)
    fa.DMD_instance.analyze.configsetting.section_FFT_spectrum_plot.Log_y_scale=False
    fa.DMD_instance.analyze.FFT_spectrum_plot()  

def init_analyze(fa:FolderAnalysis,func_logger):
    #fa.DMD_instance.start_analyze()
    fa.DMD_instance.analyze.configfile_path=fa.current_folder+'/DMDana.ini'        

#Read FFT-spectrum-plot-summary.csv to extact the DC current of 3 directions\

def read_FFT_spectrum_plot_summary(fa:FolderAnalysis,func_logger):
    strlist=["Cutoff",
            "FFT_integral_start_time_fs",
            "FFT_integral_end_time_fs",
            "Window_type",
            "FFT(jx_tot)(0)",
            "FFT(jx_d)(0)",
            "FFT(jx_od)(0)",
            "jx_tot_mean",
            "FFT(jy_tot)(0)",
            "FFT(jy_d)(0)",
            "FFT(jy_od)(0)",
            "jy_tot_mean",
            "FFT(jz_tot)(0)",
            "FFT(jz_d)(0)",
            "FFT(jz_od)(0)",
            "jz_tot_mean"]
    typelist=[float]*len(strlist)
    typelist[3]=str
    vallist=read_text_from_file('FFT-spectrum-plot-summary.csv',strlist,[1]*len(strlist),stop_at_first_find=False,dtypelist=typelist,sep=',')
    valdict=dict(zip(strlist,vallist))
    fa.df.loc[fa.index,'DC_z']=valdict["FFT(jz_tot)(0)"]
    fa.df.loc[fa.index,'DC_y']=valdict["FFT(jy_tot)(0)"]
    fa.df.loc[fa.index,'DC_x']=valdict["FFT(jx_tot)(0)"]
    fa.df.loc[fa.index,'DC_diag_z']=valdict["FFT(jz_d)(0)"]
    fa.df.loc[fa.index,'DC_diag_y']=valdict["FFT(jy_d)(0)"]
    fa.df.loc[fa.index,'DC_diag_x']=valdict["FFT(jx_d)(0)"]
    fa.df.loc[fa.index,'DC_offdiag_z']=valdict["FFT(jz_od)(0)"]
    fa.df.loc[fa.index,'DC_offdiag_y']=valdict["FFT(jy_od)(0)"]
    fa.df.loc[fa.index,'DC_offdiag_x']=valdict["FFT(jx_od)(0)"]        
    fa.df.loc[fa.index,'Cutoff_time_for_current_and_spectrum_calculation']=valdict["Cutoff"]

def get_Boltzfitted(fa:FolderAnalysis,func_logger):
    find=False
    with open('./analyze_folder_%d.log'%(fa.index)) as file:
        for line in file:
            if 'Boltzmann Distribution t(fs)' in line:
                mu=line.split()[13]
                T=line.split()[15]
                find=True
    if find:                
        mu_Boltz,T_Boltz= float(mu),float(T)
    else:
        mu_Boltz,T_Boltz= None,None
    fa.df.loc[fa.index,'mu_Boltz(eV)']=mu_Boltz
    fa.df.loc[fa.index,'T_Boltz(K)']=T_Boltz

def tell_system(fa:FolderAnalysis,func_logger):
    if 'GaAs' in fa.DMD_instance.DMD_folder:
        fa.df.loc[fa.index,'System']='GaAs'
    elif 'RhSi' in fa.DMD_instance.DMD_folder:
        fa.df.loc[fa.index,'System']='RhSi'
    elif 'GeS' in fa.DMD_instance.DMD_folder or 'ges' in fa.DMD_instance.DMD_folder:
        fa.df.loc[fa.index,'System']='GeS'

def get_conduction_change(fa:FolderAnalysis,func_logger):
    init_analyze(fa,func_logger)
    fa.DMD_instance.analyze.configsetting.section_occup_time.t_max=-1
    config_result=fa.DMD_instance.analyze.config_result.occup_time
    occup_timestep_for_all_files=config_result.occup_timestep_for_all_files
    filelist_step=int(np.round(500/occup_timestep_for_all_files))
    fa.DMD_instance.analyze.configsetting.section_occup_time.filelist_step=filelist_step
    occup_t_tot=config_result.occup_t_tot
    t_max_for_occup_time=-1 if occup_t_tot<2502 else 2502
    fa.DMD_instance.analyze.configsetting.section_occup_time.t_max=t_max_for_occup_time
    occup_t_tot_update=fa.DMD_instance.analyze.config_result.occup_time.occup_t_tot
    for i in fa.DMD_instance.analyze.config_result.occup_time.occup_selected_files:
        if DMDparser.occupation_file_class(i).time_fs< occup_t_tot_update+fa.DMD_instance.analyze.config_result.occup_time.occup_timestep_for_selected_file_ps*1000/2:
            lastone=i
    occupationfile=DMDparser.occupation_file_class(lastone)
    time_fs_to_eval_occupa=occupationfile.time_fs
    EcMin_eV=max(fa.DMD_instance.mu_eV, fa.DMD_instance.lindblad_init.energy.EcMin_eV)
    EvMax_eV=min(fa.DMD_instance.mu_eV, fa.DMD_instance.lindblad_init.energy.EvMax_eV)
    all_energy_list=occupationfile.data_eV[:,0]
    
    conduction_band_occupation=occupationfile.data_eV[all_energy_list>=EcMin_eV,1]
    conduction_band_energy_list=occupationfile.data_eV[all_energy_list>=EcMin_eV,0]
    valence_band_occupation=occupationfile.data_eV[all_energy_list<=EvMax_eV,1]
    valence_band_energy_list=occupationfile.data_eV[all_energy_list<=EvMax_eV,0]
    Conduction_band_Max_Change=np.max(np.abs(conduction_band_occupation-fermi(fa.DMD_instance.temperature_K*Kelvin,fa.DMD_instance.mu_eV*eV,conduction_band_energy_list*eV)))
    Valence_band_Max_Change=np.max(np.abs(valence_band_occupation-fermi(fa.DMD_instance.temperature_K*Kelvin,fa.DMD_instance.mu_eV*eV,valence_band_energy_list*eV)))
    fa.df.loc[fa.index,'Max_Occupation_change_Conduction']=Conduction_band_Max_Change
    fa.df.loc[fa.index,'Max_Occupation_change_Valence']=Valence_band_Max_Change
    fa.df.loc[fa.index,'time_fs_to_evaluate_occupation_change']=time_fs_to_eval_occupa


@dataclass    
class logic_func():
    file_prefix:str
    func_list:list= field(default_factory=list)

analyze=logic_func(file_prefix='analyze',
                   func_list=[ init_analyze,
                        current_plot,
                        FFT_spectrum_plot_log,
                        FFT_spectrum_plot_linear,
                        occup_time,
                        #occup_deriv,
                        occup_time_short_range_for_better_fit])

summary=logic_func(file_prefix='summary',
                func_list=[read_lindblad_out,
                get_Scat_State,
                tell_system,
                parse_param_in,
                get_step_and_time,
                get_T_and_mu,
                get_Boltzfitted,
                read_FFT_spectrum_plot_summary,
                get_conduction_change])
