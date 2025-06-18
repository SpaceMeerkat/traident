import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def getChunk(dataset, ra_rank, dec_rank, blockSize=64):
    dec = (dec_rank * blockSize, dec_rank * blockSize + 2*blockSize)
    ra = (ra_rank * blockSize, ra_rank * blockSize + 2*blockSize)
    chunk = cube.isel(**{'DEC--SIN': slice(dec[0], dec[1]), 'RA---SIN': slice(ra[0], ra[1])}).data
    return chunk

# def getChunkIndexed(dataset, ra, dec, blockSize=64):
#     dec = (dec - blockSize, dec + blockSize)
#     ra = (ra - blockSize, ra + blockSize)
#     chunk = dataset.isel(**{'DEC--SIN': slice(dec[0], dec[1]), 'RA---SIN': slice(ra[0], ra[1])}).data
#     return chunk

def getChunkIndexed(dataset, ra, dec, blockSize=64):
    dec = (dec, dec + blockSize)
    ra = (ra, ra + blockSize)
    chunk = dataset.isel(**{'Y': slice(dec[0], dec[1]), 'X': slice(ra[0], ra[1])}).compute().values
    return chunk

def normal_sample(xarray, channel, amplitude, duration):
    ### Creates a normal distribution of points with no scatter
    a = 1 / (duration*np.sqrt(2*np.pi))
    b = np.exp(-0.5 * (((xarray-channel)/duration)**2))
    norm = a * b
    norm = amplitude * norm / norm.max()
    return norm

def inject_fake_periodic_signal(cube, row, column, phase, amplitude, wavelength):
    ### Injects a fake sinusoidal signal with amplitude: N * pixel rms, with a certain wavelength and phase. 
    pixel = cube[:, row, column] # GET the current pixel timeseries
    rms = np.sqrt((pixel*pixel).sum()/cube.shape[0]) # Calculate the rms of the pixel 
    xarray = np.linspace(0,2*np.pi,cube.shape[0])
    freq = cube.shape[0] / wavelength
    signal = amplitude * rms * np.sin(phase + xarray*freq)
    signal[signal < 0] = 0 # change the sin wave into an only positive signal
    signal[pixel == 0] = 0 # Remove signal where original cube == 0
    cube[:, row, column] += signal # Inject the signal into the cube
    return cube

def inject_periodic_gaussian_signal(cube, row, column, phase, amplitude, wavelength, width=8):
    pixel = cube[:, row, column] # GET the current pixel timeseries
    rms = np.sqrt((pixel*pixel).sum()/cube.shape[0]) # Calculate the rms of the pixel 
    n_frames = len(pixel)
    xarray = np.arange(n_frames)

    freq = 2 * np.pi / wavelength
    center_offset = (phase / (2 * np.pi)) * wavelength  # convert phase to frame offset

    # Calculate all center positions of Gaussians
    centers = np.arange(-wavelength, n_frames + wavelength, wavelength) + center_offset

    # Width of each Gaussian accounting for sigma not equating to wing width and seconds conversion: width/8/6
    sigma = width/48

    # Sum of all Gaussian peaks
    signal = np.zeros(n_frames)
    for c in centers:
        signal += np.exp(-0.5 * ((xarray - c) / sigma) ** 2)

    # Normalize and scale
    signal *= (amplitude * rms)
    signal[pixel == 0] = 0

    cube[:, row, column] += signal

    return cube

def inject_fake_signal(cube, row, column, channel, amplitude, duration):
    ### Injects a fake normal signal with amplitude: N * pixel rms, at a certain channel and sigme=duration. 
    xarray = np.arange(cube.shape[0]) # Make an empty array of zeros
    pixel = cube[:, row, column] # GET the current pixel timeseries
    rms = np.sqrt((pixel*pixel).sum()/cube.shape[0]) # Calculate the rms of the pixel 
    signal = normal_sample(xarray, channel, amplitude*rms, duration) # Make the clean signal
    signal[pixel == 0] = 0 # Remove signal where original cube == 0
    signal[:channel-5*duration] = 0 # clipping signals wider than a real burst
    signal[channel+5*duration] = 0 # clipping signals wider than a real burst
    cube[:, row, column] += signal # Inject the signal into the cube
    return cube

# Function to compute std for a given time slice
def compute_std(t):
    idx = np.clip(base_idx + t, 0, time_steps - 1)
    selected_data = flux_grad_raw[idx]
    
    # Ensure there are valid (non-NaN) values
    if np.isnan(selected_data).all():
        return np.full((height, width), np.nan)  # Return NaN array if no valid values
    
    return np.nanstd(selected_data, axis=0, ddof=1)  # Use ddof=1 for unbiased std

def fft_autocorr_chunk(chunk):
    # Zero-pad to 2*T for full autocorr
    T, H, W = chunk.shape
    n = 2 * T
    # FFT along time axis
    fft_chunk = np.fft.fft(chunk, n=n, axis=0)
    # Power spectrum
    power = np.abs(fft_chunk)**2
    # IFFT to get autocorrelation
    ac_full = np.fft.ifft(power, axis=0).real
    # Normalize and return only positive lags
    return ac_full[:T]

# def flux_grad_input(chunk):

#     flux_grad_raw = np.diff(chunk, 0) 
#     flux_grad_raw[flux_grad_raw == 0] = np.nan

#     pad = 15
#     time_steps, height, width = flux_grad_raw.shape
#     base_idx = np.arange(-pad, pad, 1)

#     # Use multiprocessing to parallelize computation
#     with mp.Pool(processes=mp.cpu_count()) as pool:
#         results = list(tqdm(pool.imap(compute_std, range(time_steps)), total=time_steps))

#     flux_grad_local = np.array(results)
#     flux_grad = np.divide(flux_grad_raw, flux_grad_local, where=~np.isnan(flux_grad_local))
#     return flux_grad