from ..lib.DMDparser import DMDana_ini_Class,config_base,config_current,config_occup,allfuncname,libpath
# import these to here for compatibility with old code of other files which need these

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
