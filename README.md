## Installation

1. download this.
2. run ```export PATH="{download-path}/DMDana:$PATH"```, where {download-path} is the path where you download DMDana.
3. install relevant python libraries whenever it output lack of dependencies.
4. Optional: to support CLI auto-completion run ```activate-global-python-argcomplete --user``` (after you install argcomplete by ```pip install argcomplete```), and restart your shell or run `source ~/.bashrc`
## Usage
1. run ```cd {DMD-folder}```, where ```{DMD-folder}``` is your DMD folder path
2. run ```DMDana init``` to create ```DMDana.ini``` in this folder, ```DMDana.ini``` is the configuration file of DMDana.
3. modify ```DMDana.ini``` in your DMD folder to change DMDana parameters
4. run ```DMDana {command}``` to analyze the results, where ```{command}``` is the command you want to use. Different commands supported are listed in the "Command Supported" section.

Hint: DMDana support CLI auto-completion, try press ```TAB``` button after you type ```DMDana``` in your shell.

## Help and documents
1. run ```DMDana -h``` to see the help information about CLI input
2. see DMDana.ini to see its comments on different parameters of DMDana.

## Command Supported
| command | feature |
| ---- | ---- |
|init|initialize the DMDana.ini file if it does not exist.|
|FFT_DC_convergence_test|This plots the change of the Direct Current component calculated by different FFT-time-range and FFT-window-functions. This aims to check FFT convergence. It could also output the analysis results to files.|
|FFT_spectrum_plot|This plots FFT spectra of the DMD currents. It could also output the DC components of the current FFT-setting to files|
|current_plot|This plots the current figures|
|occup_time|This plots the occupation functions with time, namely f(E,t)  of different styles in batch.|
|occup_deriv|This plots the E-axis maximum for the time derivative of the occupation funciton f(E,t), namely (df/dt)max_in_E_axis.|

