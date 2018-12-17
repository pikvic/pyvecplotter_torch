import numpy as np

def ssim(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)

    C = np.sum(x_diff*y_diff) / np.sqrt(np.sum(np.power(x_diff,2))*np.sum(np.power(y_diff,2)))
    E = 1 - np.sum(np.abs(x_diff - y_diff)) / (np.sum(np.abs(x_diff)) + np.sum(np.abs(y_diff)))
    S = 2*x_std*y_std/ (np.power(x_std,2) + np.power(y_std,2))
    
    ssim = C * E * S
    return ssim

def ssim_matrix(area1, area2, search_radius, window_radius):

    ssim_size = search_radius*2 - window_radius*2 + 1
    window_size = window_radius*2+1
    padding = search_radius - window_radius
    
    ssim_matrix = np.zeros((ssim_size, ssim_size))
    win1 = area1[padding : padding + window_size, padding : padding + window_size]
    
    for y in range(ssim_size):
        for x in range(ssim_size):
            win2 = area2[0+y:window_size+y, 0+x:window_size+x]
            ssim_matrix[y, x] = ssim(win1, win2)
      
    return ssim_matrix

def process_point(data1, data2, x0, y0, search_radius, window_radius):

    area1 = data1[y0-search_radius:y0+search_radius+1, x0-search_radius:x0+search_radius+1]
    area2 = data2[y0-search_radius:y0+search_radius+1, x0-search_radius:x0+search_radius+1]

    ssim12 = ssim_matrix(area1, area2, search_radius, window_radius)
    ssim11 = ssim_matrix(area1, area1, search_radius, window_radius)

    ssim_max = np.max(ssim12)
    i, j = np.unravel_index(ssim12.argmax(), ssim12.shape) 
    y1 = y0 - search_radius + window_radius + i
    x1 = x0 - search_radius + window_radius + j

    area2 = data2[y1-search_radius:y1+search_radius+1, x1-search_radius:x1+search_radius+1]
    ssim22 = ssim_matrix(area2, area2, search_radius, window_radius)

    id1_max = np.argwhere(ssim11 == np.max(ssim11))
    id2_max = np.argwhere(ssim22 == np.max(ssim22))
    id1 = np.argwhere(ssim11 > ssim_max)
    id2 = np.argwhere(ssim22 > ssim_max)
    r1 = np.max(np.sqrt(np.einsum('ij,ij->i', id1 - id1_max, id1-id1_max)))
    r2 = np.max(np.sqrt(np.einsum('ij,ij->i', id2 - id2_max, id2-id2_max)))
 
    r_max = max(r1, r2)
    #assim = pix_size_km * r_max * 1000 / dt
    #v = pix_size_km * np.sqrt(np.power(x1 - x0, 2) + np.power(y1-y0, 2))*1000/dt
   # lon0 = mapper.lon(column=x0)
   # lat0 = mapper.lat(scan=y0)
   # lon1 = mapper.lon(column=x1)
   # lat1 = mapper.lat(scan=y1)

    return (x0, y0, x1, y1, ssim_max, r_max)


def run_algorithm(data1, data2, config):

    left = config['left']
    top = config['top']
    step = config['step']
    nx = config['nx']
    ny = config['ny']
    search_radius_x = config['search_radius_x']
    search_radius_y = config['search_radius_y']
    window_radius_x = config['window_radius_x']
    window_radius_y = config['window_radius_y']

    points = [(x, y) for x in range(left, left + step*nx, step) for y in range(top, top + step*ny, step)]

    vectors = []
    
    print(f'Running algorighm with parameters:\n')
    print(f'search_radius_x={search_radius_x} search_radius_y={search_radius_y} window_radius_x={window_radius_x} window_radius_y={window_radius_y}')
    print(f'left={left} top={top} step={step} nx={nx} ny={ny}')

    for point_num, (x0, y0) in enumerate(points):
        print(f'Processing point  {point_num+1}/{len(points)} : ({x0}, {y0})...')
        vector = process_point(data1, data2, x0, y0, search_radius_x, window_radius_x)
        vectors.append(vector)

    return vectors
    