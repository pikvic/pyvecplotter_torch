import time
import numpy as np
import vecplotter_numpy as vecnp
import vecplotter_torch as vecth

config = {
    'left': 50,
    'top': 50,
    'step': 5,
    'nx': 2,
    'ny': 2,
    'search_radius_x': 40,
    'search_radius_y': 40,
    'window_radius_x': 20,
    'window_radius_y': 20,
}

def generate_data(N=150, search_radius=30, window_radius=10):

    data1 = np.random.randint(0, high=100, size=(N, N)).astype(np.float)
    data2 = np.random.randint(0, high=100, size=(N, N)).astype(np.float)
    
    return data1, data2

if __name__ == "__main__":
    
    N = 150
    data1, data2 = generate_data(N, config['search_radius_x'], config['window_radius_x'])

    t = time.time()
    vecnp.run_algorithm(data1, data2, config)
    print(f'Numpy version - time elapsed: {time.time() - t:.2} seconds')

    t = time.time()
    vecth.run_algorithm(data1, data2, config)
    print(f'PyTorch version - time elapsed: {time.time() - t:.2} seconds')