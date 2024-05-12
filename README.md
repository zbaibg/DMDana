## Installation

Requirement: Python version >3.2

1. Make sure git it installed in your system. (to support logging git-hash during DMDana running)
2. Download this. Run ```cd DMDana```,  ```pip install .```
. If you want skip rerun pip install for code upgrade in the future (upgrade through ```git pull```), you can use "editable install mode" by ```pip install -e .```.
3. Optional: to support CLI auto-completion run ```activate-global-python-argcomplete --user```, and restart your shell. (After restarting, if it does not work properly, consider manually running ```source ~/.bash_completion``` and add this command to your ```~/.bashrc```)
## Usage
1. run ```cd {DMD-folder}```, where ```{DMD-folder}``` is your DMD folder path
2. run ```python -m DMDana.do init``` to create ```DMDana.ini``` in this folder, ```DMDana.ini``` is the configuration file of DMDana.
3. modify ```DMDana.ini``` in your DMD folder to change DMDana parameters
4. run ```python -m DMDana.do {command}``` to analyze the results, where ```{command}``` is the command you want to use. Different commands supported are listed in the "Command Supported" section.

Hint: DMDana support CLI auto-completion, try press ```TAB``` button after you type ```python -m DMDana.do``` in your shell.

## Help and documents
1. run ```python -m DMDana.do -h``` to see the help information about CLI input
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

## Demo for Plotting and Data Outputted
Using DMDana, you can easily get the following figures or data with custom setting.
![pic](/pic-demo/j.png)
![pic](/pic-demo/occup-time1.png)
![pic](/pic-demo/occup-time2.png)
![pic](/pic-demo/occup-time4.png)
![pic](/pic-demo/occup-time3.png)
![pic](/pic-demo/fft.png)
![pic](/pic-demo/fft-test.png)
![pic](/pic-demo/fft-data.png)
