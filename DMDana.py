#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""_summary_
Main executable file for DMDana
Please run activate-global-python-argcomplete --user in the terminal to enable the auto-completion of the command line arguments.
"""
import argparse
import argcomplete
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
if args.command == "FFT_DC_convergence_test":
    import FFT_DC_convergence_test
    FFT_DC_convergence_test.do()
elif args.command == "FFT_spectrum_plot":
    import FFT_spectrum_plot
    FFT_spectrum_plot.do()
elif args.command == "current_plot":
    import current_plot
    current_plot.do()
elif args.command == "occup_deriv":
    import occup_deriv
    occup_deriv.do()
elif args.command == "occup_time":
    import occup_time
    occup_time.do()
elif args.command == "init":
    import sys,shutil,os
    if(not os.path.isfile('DMDana.ini')):
        shutil.copyfile(sys.path[0]+'/DMDana.ini','./DMDana.ini')
    else:
        print("DMDana.ini already exists.")
elif args.command == None:
    parser.print_help()