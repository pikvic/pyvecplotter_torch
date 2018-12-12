import time
import numpy as np
import vecplotter_numpy as vecnp
import vecplotter_torch as vecth
import vecplotter_torch_refactored as vecthref
#from memory_profiler import profile

config = {
    'left': 150,
    'top': 150,
    'step': 5,
    'nx': 4,
    'ny': 4,
    'search_radius_x': 50,
    'search_radius_y': 50,
    'window_radius_x': 20,
    'window_radius_y': 20,
}

def generate_data(N=150, search_radius=30, window_radius=10):

    data1 = np.random.randint(0, high=100, size=(N, N)).astype(np.float)
    data2 = np.random.randint(0, high=100, size=(N, N)).astype(np.float)
    
    return data1, data2

def estimate_memory_footprints(npoints=1, search_radius=100, window_radius=30, element_size=8):
    window_size = window_radius*2 + 1
    search_size = search_radius*2 + 1
    window_numel = window_radius**2
    ssim_size = search_size - window_size + 1
    ssim_numel = ssim_size**2
    total_numel = ssim_numel * window_numel
    total_size = total_numel * element_size * npoints
    result_numel = ssim_numel * npoints
    result_size = result_numel * element_size
    print(f'Estimated size: {total_numel} elements, {total_size / 1024 / 1024 :.2f} MB')
    print(f'Estimated size of result: {result_numel} elements, {result_size / 1024 / 1024 :.2f} MB')

#@profile
def main():
    N = 250
    data1, data2 = generate_data(N, config['search_radius_x'], config['window_radius_x'])

    t = time.time()
    vecnp.run_algorithm(data1, data2, config)
    print(f'Numpy version - time elapsed: {time.time() - t:.2} seconds')

    t = time.time()
    vecth.run_algorithm(data1, data2, config)
    print(f'PyTorch version - time elapsed: {time.time() - t:.2} seconds')

    t = time.time()
    vecthref.run_algorithm(data1, data2, config)
    print(f'PyTorch version refactored - time elapsed: {time.time() - t:.2} seconds')

if __name__ == "__main__":
    main()
    estimate_memory_footprints()
    