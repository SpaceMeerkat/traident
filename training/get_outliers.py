from mpi4py import MPI
import numpy as np
import xarray as xr
import os

# --- Parameters ---
zarrpath = "/idia/users/jdawson/transient/traident/datasets/obs-l2-qc2/cubes/cube-1.zarr"
blockSize = 512
output_path = "training_output/outlier_timeseries.npy"
coords_path = "training_output/outlier_coordinates.npy"

# MPI setup
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Loading zarr dataset on rank: {rank}")
xrdata = xr.open_dataset(zarrpath, engine='zarr', chunks={})
cube = xrdata.cube
T, Y, X = cube.sizes['TIME'], cube.sizes['Y'], cube.sizes['X']

print(f"Loading outlier coordinates on rank: {rank}")
outliers = np.load('training_output/outlier_positions.npy')  # shape (2, N)
outlier_coords = list(zip(outliers[0], outliers[1]))  # (x, y) tuples

# Determine chunks
X_starts = list(range(0, X, blockSize))
Y_starts = list(range(0, Y, blockSize))
all_chunks = [(x, y) for x in X_starts for y in Y_starts]

# Distribute chunks
my_chunks = [chunk for i, chunk in enumerate(all_chunks) if i % size == rank]
print(f"[Rank {rank}] Assigned {len(my_chunks)} chunks.")

my_data = []
my_coords = []

for idx, (x0, y0) in enumerate(my_chunks):
    x1 = min(x0 + blockSize, X)
    y1 = min(y0 + blockSize, Y)
    print(f"[Rank {rank}] Processing chunk {idx + 1}/{len(my_chunks)} at ({x0}:{x1}, {y0}:{y1})")

    chunk_outliers = [
        (x, y) for x, y in outlier_coords if x0 <= x < x1 and y0 <= y < y1
    ]

    print(f"[Rank {rank}] Found {len(chunk_outliers)} outliers in chunk.")

    for (x, y) in chunk_outliers:
        try:
            data = cube.isel(STOKES=0, X=x, Y=y).values  # shape: (T,)
            my_data.append(data)
            my_coords.append((x, y))
        except Exception as e:
            print(f"[Rank {rank}] Failed to read (x={x}, y={y}): {e}")

print(f"[Rank {rank}] Finished extracting {len(my_data)} time series.")

# Gather all to root
all_data = comm.gather(my_data, root=0)
all_coords = comm.gather(my_coords, root=0)

# Root: concatenate and save
if rank == 0:
    print("[Rank 0] Gathering data from all ranks...")
    flat_data = [arr for sublist in all_data for arr in sublist]
    flat_coords = [coord for sublist in all_coords for coord in sublist]

    combined = np.stack(flat_data, axis=0)
    coords_array = np.array(flat_coords, dtype=np.int32)  # shape: (N, 2)

    print(f"[Rank 0] Final data shape: {combined.shape}")
    print(f"[Rank 0] Final coords shape: {coords_array.shape}")

    np.save(output_path, combined)
    np.save(coords_path, coords_array)
    print("[Rank 0] Data and coordinates saved successfully.")
