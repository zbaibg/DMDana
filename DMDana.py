#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
import argparse
import argcomplete
parser = argparse.ArgumentParser(description="DMD post-analysis script collection")  
subparsers = parser.add_subparsers(dest="command")
subparsers.add_parser("FFT_DC_convergence_test", help="This script plots the DC components of FFT results with different parameters in batch. It also output the data to text.")
subparsers.add_parser("FFT_spectrum_plot",help="This script plots FFT spectra with different parameters in batch.")
subparsers.add_parser("current_plot", help="This script plots the current figures")
subparsers.add_parser("occup_deriv", help="This script plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.")
subparsers.add_parser("occup_time", help="This script plot the occupation functions with time, f(E,t).")
argcomplete.autocomplete(parser)
args=parser.parse_args()
if args.command == "FFT_DC_convergence_test":
    import FFT_DC_convergence_test
    FFT_DC_convergence_test.do()
if args.command == "FFT_spectrum_plot":
    import FFT_spectrum_plot
    FFT_spectrum_plot.do()
if args.command == "current_plot":
    import current_plot
    current_plot.do()
if args.command == "occup_deriv":
    import occup_deriv
    occup_deriv.do()
if args.command == "occup_time":
    import occup_time
    occup_time.do()