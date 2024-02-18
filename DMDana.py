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
    subparsers.add_parser("FFT_DC_convergence_test", help="This plots the DC components of FFT results with different parameters in batch. It also output the data to text.")
    subparsers.add_parser("FFT_spectrum_plot",help="This plots FFT spectra with different parameters in batch.")
    subparsers.add_parser("current_plot", help="This plots the current figures")
    subparsers.add_parser("occup_deriv", help="This plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.")
    subparsers.add_parser("occup_time", help="This plot the occupation functions with time, f(E,t).")
    argcomplete.autocomplete(parser)
    args=parser.parse_args()
    if args.command==None:
        parser.print_help()
        exit()
    #CLI parameter solved. Now we can start the main program.

    #init command: initialize the DMDana.ini file if it does not exist.
    if args.command == "init":
        import sys,shutil,os
        if(not os.path.isfile('DMDana.ini')):
            shutil.copyfile(sys.path[0]+'/DMDana_default.ini','./DMDana.ini')
        else:
            print("DMDana.ini already exists.")
        exit()
        
    #other commands    
    from config import configclass
    import logging
    import global_variable    
    funcname=args.command
    logging.basicConfig(
        level=logging.INFO,
        filename='DMDana_'+funcname+'.log',
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        filemode='w',)
    global_variable.config=configclass(funcname.replace('_','-'))
    if funcname == "FFT_DC_convergence_test":
        import FFT_DC_convergence_test
        FFT_DC_convergence_test.do()
    elif funcname == "FFT_spectrum_plot":
        import FFT_spectrum_plot
        FFT_spectrum_plot.do()
    elif funcname == "current_plot":
        import current_plot
        current_plot.do()
    elif funcname == "occup_deriv":
        import occup_deriv
        occup_deriv.do()
    elif funcname == "occup_time":
        import occup_time
        occup_time.do()

            
    logging.info('Done successfully!')