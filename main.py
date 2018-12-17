import time
import numpy as np
import vecplotter_numpy as vecnp
import vecplotter_torch as vecth
import vecplotter_torch_refactored as vecthref
import torch
#from memory_profiler import profile

config = {
    'left': 150,
    'top': 150,
    'step': 5,
    'nx': 4,
    'ny': 4,
    'search_radius_x': 30,
    'search_radius_y': 30,
    'window_radius_x': 10,
    'window_radius_y': 10,
    'element_size': 8,
    'memory_limit': 3,
    'device': 'auto',
}

def generate_data(N=150, search_radius=30, window_radius=10):

    data1 = np.random.randint(0, high=100, size=(N, N))#.astype(np.float)
    data2 = np.random.randint(0, high=100, size=(N, N))#.astype(np.float)
    
    return data1, data2

def get_batch_size(memory_limit_gb, config):
    npoints = config['nx'] * config['ny']
    window_radius = config['window_radius_x']
    search_radius = config['search_radius_x']

    GB = 1024 ** 3
    MB = 1024 ** 2

    element_size = 8
    window_size = window_radius * 2 + 1
    search_size = search_radius * 2 + 1
    window_numel = window_size ** 2
    ssim_size = search_size - window_size + 1
    ssim_numel = ssim_size ** 2
    point_numel = ssim_numel * window_numel
    point_size = point_numel * element_size * 3
    total_numel = point_numel * npoints
    total_size = point_size * npoints
    memory_limit = memory_limit_gb * GB
    
    
    batch_size = memory_limit // point_size
    nbatches = total_size // memory_limit
    total_limit = nbatches * memory_limit /  MB
    
    print(f'{total_numel} Elements / {total_size / MB } MB / Limit = {memory_limit / MB } MB')
    print(f'nbatches = {nbatches} / Total_Limit = {total_limit} MB')
    print(f'Point_size = {point_size / MB} MB / Batch_size = {batch_size}')

    return batch_size

    
def estimate_memory_footprints(npoints=1, search_radius=100, window_radius=30, element_size=8):
    window_size = window_radius * 2 + 1
    search_size = search_radius * 2 + 1
    window_numel = window_size ** 2
    ssim_size = search_size - window_size + 1
    ssim_numel = ssim_size ** 2
    total_numel = ssim_numel * window_numel
    total_size = total_numel * element_size * npoints
    result_numel = ssim_numel * npoints
    result_size = result_numel * element_size
    print(f'{[npoints, ssim_numel, window_numel]} - {total_numel} - {total_size / 1024 / 1024 :.2f} MB')
    print(f'Estimated size: {total_numel} elements, {total_size / 1024 / 1024 :.2f} MB')
    print(f'Estimated size of result: {result_numel} elements, {result_size / 1024 / 1024 :.2f} MB')

#@profile
def main():
    N = 350
    data1, data2 = generate_data(N, config['search_radius_x'], config['window_radius_x'])

    # t = time.time()
    # vectors_np = vecnp.run_algorithm(data1, data2, config)
    # print(f'Numpy version - time elapsed: {time.time() - t:.2f} seconds')

    # t = time.time()
    # vecth.run_algorithm(data1, data2, config)
    # print(f'PyTorch version - time elapsed: {time.time() - t:.2} seconds')

    t = time.time()
    vectors_th_ref = vecthref.run_algorithm(data1, data2, config)
    print(f'PyTorch version refactored - time elapsed: {time.time() - t:.2f} seconds')

    # for v1, v2 in zip(vectors_np, vectors_th_ref):
    #     print(v1, v2)

if __name__ == "__main__":
    estimate_memory_footprints(16, 60, 20)
    
    config['batch_size'] = get_batch_size(3, config)
    print(config['batch_size'])
    main()
  
    print(torch.cuda.get_device_properties(0).total_memory - torch.cuda.get_device_properties(0).total_memory*0.25)