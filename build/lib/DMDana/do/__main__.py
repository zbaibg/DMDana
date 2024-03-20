#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""_summary_
Main executable file for DMDana
Please run activate-global-python-argcomplete --user in the terminal to enable the auto-completion of the command line arguments.
"""
import argparse
import argcomplete

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DMD post-analysis script collection")  
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("init",help="initialize the DMDana.ini file if it does not exist.")
    subparsers.add_parser("clean_output",help="clean all output of DMDana in this folder")
    subparsers.add_parser("clean_input",help="clean DMDana.ini in this folder")
    subparsers.add_parser("FFT_DC_convergence_test", help="This plots the change of the Direct Current component calculated by different FFT-time-range and FFT-window-functions. This aims to check FFT convergence. It could also output the analysis results to files.")
    subparsers.add_parser("FFT_spectrum_plot",help="This plots FFT spectra of the DMD currents. It could also output the DC components of the current FFT-setting to files")
    subparsers.add_parser("current_plot", help="This plots the current figures")
    subparsers.add_parser("occup_time", help="This plots the occupation functions with time, namely f(E,t)  of different styles in batch.")
    subparsers.add_parser("occup_deriv", help="This plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.")
    argcomplete.autocomplete(parser)
    args=parser.parse_args()
    if args.command==None:
        parser.print_help()
        exit()
    #CLI parameter solved. Now we can start the main program.

    #init command: initialize the DMDana.ini file if it does not exist.
    if args.command == "init":
        import sys,shutil,os
        from .config import libpath
        if(not os.path.isfile('DMDana.ini')):
            shutil.copyfile(libpath+'/DMDana/do/DMDana_default.ini','./DMDana.ini')
        else:
            print("DMDana.ini already exists.")
        exit()
    if args.command == "clean_output":
        import os
        os.system('rm -f DMDana_*.log')
        os.system('rm -f *.png')
        os.system('rm -f FFT*')
        # need to be speific about each file in the future
        exit()    
    if args.command == "clean_input":
        import os
        os.system('rm -f DMDana.ini')
        exit()  
    #other commands    
    from .config import autoconfig
    import logging
    from . import global_variable    
    funcname=args.command
    logging.basicConfig(
        level=logging.INFO,
        filename='DMDana_'+funcname+'.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='a',)
    config=autoconfig(funcname.replace('_','-'))
    global_variable.config=config
    if funcname == "FFT_DC_convergence_test":
        from . import FFT_DC_convergence_test
        FFT_DC_convergence_test.do()
    elif funcname == "FFT_spectrum_plot":
        from . import FFT_spectrum_plot
        FFT_spectrum_plot.do()
    elif funcname == "current_plot":
        from . import current_plot
        current_plot.do(config)
    elif funcname == "occup_deriv":
        from . import occup_deriv
        occup_deriv.do()
    elif funcname == "occup_time":
        from . import occup_time
        occup_time.do()

            
    logging.info('Done successfully!')