import numpy as np
import scipy.signal.windows as sgl

from . import constant as const


# funciton which performs FFT, 
# shifts frequency bins to only plot positive frequencies, 
# changes bins to physical units (eV), applies window to time domain data, 
# and returns a normalized FFT
def fft_of_j(j_t, cutoff,Window_type):
    dt = j_t[1,0] - j_t[0,0]
    N_jt = j_t[cutoff:,1].shape[0]
    freq_bins = np.fft.fftfreq(N_jt, dt)*(2.0*np.pi*const.Hatree_to_eV)
    shifted_freq_bins = freq_bins[:len(freq_bins)//2]

    if Window_type=='Flattop':#Rectangular, Flattop, Hann, Hamming 
        window = sgl.flattop(N_jt, sym=False)
    elif Window_type=='Hann':
        window = sgl.hann(N_jt, sym=False)
    elif Window_type=='Hamming':
        window = sgl.hamming(N_jt,sym=False)
    elif Window_type=='Rectangular':
        window= 1
    else:
        raise ValueError('Window type is not supported')
    win_jt = window*j_t[cutoff:,1]
    
    fft = np.fft.fft(win_jt)/np.mean(window)
    shifted_fft = fft[:N_jt//2]
    return shifted_freq_bins, (1/N_jt)*(shifted_fft)