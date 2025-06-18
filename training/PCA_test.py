### 3rd party packages
from mpi4py import MPI
import numpy as np
import xarray as xr
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

def getChunkIndexed(dataset, ra, dec, blockSize=512):
    dec = (dec, dec + blockSize)
    ra = (ra, ra + blockSize)
    chunk = dataset.isel(**{'Y': slice(dec[0], dec[1]), 'X': slice(ra[0], ra[1])}).compute().values
    return chunk

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

def runPCAAnomalies(inputs, n_components=32):
    pca = PCA(n_components=n_components)
    pca.fit(inputs)
    PCA_dims = pca.transform(inputs)
    return pca, PCA_dims

def normal_sample(xarray, channel, amplitude, duration):
    ### Creates a normal distribution of points with no scatter
    a = 1 / (duration*np.sqrt(2*np.pi))
    b = np.exp(-0.5 * (((xarray-channel)/duration)**2))
    norm = a * b
    norm = amplitude * norm / norm.max()
    return norm

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

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Dataset path
# zarrpath = "/idia/users/jdawson/transient/traident/datasets/obs-l2-qc2/cubes/cube-1.zarr"
zarrpath = "/idia/users/jdawson/transient/traident/datasets/obs-omcen/cubes/cube-1.zarr"
xrdata = xr.open_dataset(zarrpath, engine='zarr', chunks={})
cube = xrdata.cube

blockSize = 128  # or 512 or whatever you want

# Dataset spatial dimensions
X_size = cube.sizes['X']
Y_size = cube.sizes['Y']

# Generate start indices with step=blockSize, ensuring we don't overshoot
X_starts = list(range(0, X_size, blockSize))
Y_starts = list(range(0, Y_size, blockSize))

all_chunks = []
for x in X_starts:
    for y in Y_starts:
        # Make sure block doesn't exceed dataset bounds
        if x + blockSize <= X_size and y + blockSize <= Y_size:
            all_chunks.append((x, y))
        else:
            print('Warning, the block size is larger than the data size!')
            pass  # You can handle edge blocks here if you want

all_chunks = [(x, y) for x in X_starts for y in Y_starts]

# Constants for communication
TAG_TASK = 1
TAG_DONE = 2
TAG_EXIT = 0

if rank == 0:
    # === MASTER ===
    from collections import deque
    task_queue = deque(all_chunks)
    results = []
    counter = 0

    num_workers = size - 1
    active_workers = 0

    # Initial task dispatch
    for worker in range(1, size):
        if task_queue:
            chunk_coords = task_queue.popleft()
            comm.send(chunk_coords, dest=worker, tag=TAG_TASK)
            active_workers += 1

    while active_workers > 0:
        # Receive results
        data = comm.recv(source=MPI.ANY_SOURCE, tag=TAG_DONE)
        source = data['rank']
        results.append(((data['ra'], data['dec']), data['residuals']))
        counter += 1
        print(f'Rank 0 received {counter}/{len(all_chunks)} residual blocks.')

        # Send new task or shut down worker
        if task_queue:
            chunk_coords = task_queue.popleft()
            comm.send(chunk_coords, dest=source, tag=TAG_TASK)
        else:
            comm.send(None, dest=source, tag=TAG_EXIT)
            active_workers -= 1

    # === Assemble final result ===
    res_map = np.zeros((cube.sizes['X'], cube.sizes['Y']), dtype=np.float32)
    for (ra, dec), res in results:
        res -= np.median(res)
        res /= np.std(res)
        res_map[ra:ra + blockSize, dec:dec + blockSize] = res.reshape((blockSize, blockSize))

    np.save('training_output/residuals_injected_omcen.npy', res_map)
    print("✅ All chunks processed and residuals saved.")
else:
    # === WORKERS ===
    while True:
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        if status.tag == TAG_EXIT:
            break

        ra, dec = task
        try:
            chunk = getChunkIndexed(cube, ra, dec, blockSize)[0]
            if (ra == 0) & (dec == 0):
                chunk = inject_fake_signal(chunk, 30, 100, 3000, 7, 16)
                chunk = inject_periodic_gaussian_signal(chunk, 40, 40, 0, 7, 120, width=16)
                chunk = inject_periodic_gaussian_signal(chunk, 20, 100, np.pi, 7, 120, width=8)
            autocorr_cube = fft_autocorr_chunk(chunk)
            autocorr_cube -= np.median(autocorr_cube, axis=(1, 2))[:, None, None]
            autocorr_cube[0, :, :] = 0
            autocorr_cube = autocorr_cube.reshape(autocorr_cube.shape[0], -1).T

            pca, dims = runPCAAnomalies(autocorr_cube)
            reconstructed = pca.inverse_transform(dims)
            residuals = np.abs(reconstructed - autocorr_cube).sum(1)

        except Exception as e:
            print(f"[Rank {rank}] Error on chunk ({ra}, {dec}): {e}")
            residuals = np.zeros(blockSize * blockSize)

        comm.send({
            'rank': rank,
            'ra': ra,
            'dec': dec,
            'residuals': residuals
        }, dest=0, tag=TAG_DONE)




