import time
import numpy as np
import vecplotter_numpy as vecnp
import vecplotter_torch as vecth
import vecplotter_torch_refactored as vecthref
import vecplotter_legacy as vecleg
import torch

config = {
    'left': 150,
    'top': 150,
    'step': 5,
    'nx': 2,
    'ny': 2,
    'search_radius_x': 20,
    'search_radius_y': 15,
    'window_radius_x': 10,
    'window_radius_y': 5,
    'element_size': 8,
    'memory_limit': 3,
    'device': 'auto',
    'sa_cx': 30, 
    'sa_cy': 30, 
    'rw_cx': 10,
    'rw_cy': 10,
    'legacy': False,
}

def generate_data(N=150, search_radius=30, window_radius=10):

    data1 = np.random.randint(0, high=100, size=(N, N))#.astype(np.float)
    data2 = np.random.randint(0, high=100, size=(N, N))#.astype(np.float)
    
    return data1, data2

def get_batch_size(memory_limit_gb, config):
    
    npoints = config['nx'] * config['ny']
    element_size = config['element_size']
    legacy = config['legacy']    
    if not legacy:
        window_radius_x = config['window_radius_x']
        search_radius_x = config['search_radius_x']
        window_radius_y = config['window_radius_y']
        search_radius_y = config['search_radius_y']

        window_size_x = window_radius_x * 2 + 1
        window_size_y = window_radius_y * 2 + 1
        search_size_x = search_radius_x * 2 + 1
        search_size_y = search_radius_y * 2 + 1
        
        ssim_size_x = search_size_x - window_size_x + 1
        ssim_size_y = search_size_y - window_size_y + 1
    else:
        sa_cx = config['sa_cx']
        sa_cy = config['sa_cy']
        rw_cx = config['rw_cx']
        rw_cy = config['rw_cy']
        
        sa_left = - sa_cx // 2
        sa_right = sa_cx // 2 - sa_cx
        sa_top = - sa_cy // 2
        sa_bottom = sa_cy // 2 - sa_cy
        
        ssim_size_x = sa_right - sa_left
        ssim_size_y = sa_bottom - sa_top
        
        rw_left = - rw_cx // 2
        rw_right = - rw_cx // 2 + rw_cx
        rw_top = - rw_cy // 2
        rw_bottom = - rw_cy // 2 + rw_cy
        
        window_size_x = rw_right - rw_left + 1
        window_size_y = rw_bottom - rw_top + 1

    window_numel = window_size_x * window_size_y
    ssim_numel = ssim_size_x * ssim_size_y
    point_numel = ssim_numel * window_numel
    point_size = point_numel * element_size * 3
    total_numel = point_numel * npoints
    total_size = point_size * npoints
    memory_limit = memory_limit_gb * 1024 ** 3
        
    batch_size = memory_limit // point_size
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


def main():
    N = 350
    data1, data2 = generate_data(N, config['search_radius_x'], config['window_radius_x'])

    t = time.time()
    vectors_np = vecnp.run_algorithm(data1, data2, config)
    print(f'Numpy version - time elapsed: {time.time() - t:.2f} seconds')

    # t = time.time()
    # vecth.run_algorithm(data1, data2, config)
    # print(f'PyTorch version - time elapsed: {time.time() - t:.2} seconds')

    t = time.time()
    vectors_legacy = vecleg.run_algorithm(data1, data2, config)
    print(f'Version legacy - time elapsed: {time.time() - t:.2f} seconds')

    t = time.time()
    vectors_th_ref = vecthref.run_algorithm(data1, data2, config)
    print(f'PyTorch version refactored - time elapsed: {time.time() - t:.2f} seconds')

    for i, (v1, v2, v3) in enumerate(zip(vectors_np, vectors_th_ref, vectors_legacy)):
        print(f'{i:=^61}')
        print(v1)
        print(v2)
        print(v3)

if __name__ == "__main__":
    estimate_memory_footprints(16, 60, 20)
    
    config['batch_size'] = get_batch_size(3, config)
    print(config['batch_size'])
    main()
    
  #  print(torch.cuda.get_device_properties(0).total_memory - torch.cuda.get_device_properties(0).total_memory*0.25)