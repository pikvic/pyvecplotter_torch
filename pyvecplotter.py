import numpy as np
import datetime
import lab34dtypes
import projmapper
import argparse
import configparser
import vecplotter_torch_refactored as vecthref
from lxml import etree

def readproj(fname):

    f = open(fname, 'rb')
    b0 = np.fromfile(f, dtype=lab34dtypes.b0_proj_dt, count=1)
    sizeX = b0['b0_proj_common']['pixNum'][0]
    sizeY = b0['b0_proj_common']['scanNum'][0]
    data = np.fromfile(f, dtype='int16')
    data = data.reshape(sizeY, sizeX)
    data = np.flipud(data)
    f.close()
    return b0, data

def calc_dt(b01, b02):

    year1 = int(b01['b0_common']['year'][0])
    day1 = int(b01['b0_common']['day'][0])
    seconds1 = int(b01['b0_common']['dayTime'][0]/1000)
    year2 = int(b02['b0_common']['year'][0])
    day2 = int(b02['b0_common']['day'][0])
    seconds2 = int(b02['b0_common']['dayTime'][0]/1000)

    date1 = datetime.datetime(year1, 1, 1)
    date2 = datetime.datetime(year2, 1, 1)

    delta1 = datetime.timedelta(days=day1-1, seconds=seconds1)
    date1 = date1 + delta1
    delta2 = datetime.timedelta(days=day2-1, seconds=seconds2)
    date2 = date2 + delta2
    dt = date2 - date1
    d_t = dt.seconds

    return d_t
    
def calc_pix_size_km(b0):

    RD = 180.0/3.14159265358979323846
    DR = 3.14159265358979323846/180.0
    Ea = 6378.137
    Eb = 6356.7523142
    pix_size_km = float(b0['b0_proj_common']['latRes'][0]) / 3600.0 * DR * (Ea+Eb)/2
    return pix_size_km


def parse_config(filename='config.ini'):
    
    config_dict = {}
    config = configparser.ConfigParser()
    config.read(filename)
    config_dict['search_radius_x'] = config.getint('PARAMETERS', 'search_radius_x')
    config_dict['search_radius_y'] = config.getint('PARAMETERS', 'search_radius_y')
    config_dict['window_radius_x'] = config.getint('PARAMETERS', 'window_radius_x')
    config_dict['window_radius_y'] = config.getint('PARAMETERS', 'window_radius_y')
    config_dict['ssim_lim_lower'] = config.getfloat('PARAMETERS', 'ssim_lim_lower')
    config_dict['assim_lim_upper'] = config.getfloat('PARAMETERS', 'assim_lim_upper')
    config_dict['v_lim_upper'] = config.getfloat('PARAMETERS', 'v_lim_upper')
    config_dict['alpha'] = config.getfloat('PARAMETERS', 'alpha')
    config_dict['beta'] = config.getfloat('PARAMETERS', 'beta')
    config_dict['gamma'] = config.getfloat('PARAMETERS', 'gamma')
    config_dict['left'] = config.getint('POINTS', 'left')
    config_dict['top'] = config.getint('POINTS', 'top')
    config_dict['step'] = config.getint('POINTS', 'step')
    config_dict['nx'] = config.getint('POINTS', 'nx')
    config_dict['ny'] = config.getint('POINTS', 'ny')
    print(config_dict)
    
    return config_dict

def read_config_legacy(fname):
    
    config = {}

    with open(fname, 'rt') as f:
        xml = f.read()

    tree = etree.fromstring(xml)

    config['sa_cx'] = int(tree.xpath('/config/sa_cx/text()')[0])
    config['sa_cy'] = int(tree.xpath('/config/sa_cy/text()')[0])

    iterations = tree.xpath('//iteration')
    for iteration in iterations:
        config['rw_cx'] = int(iteration.xpath('//rw_cx/text()')[0])
        config['rw_cy'] = int(iteration.xpath('//rw_cy/text()')[0])
        config['ssim_lim_lower'] = float(iteration.xpath('//SSIM_thr/text()')[0])
        config['assim_lim_upper'] = float(iteration.xpath('//aSSIM_thr/text()')[0])
        config['v_thr'] = float(iteration.xpath('//v_thr/text()')[0])
        config['C_pow'] = int(iteration.xpath('//C/text()')[0])
        config['E_pow'] = int(iteration.xpath('//E/text()')[0])
        config['S_pow'] = int(iteration.xpath('//S/text()')[0])
        
    grids = tree.xpath('//grids/regular5')
    for grid in grids:
        config['left'] = int(grid.xpath('//left/text()')[0])
        config['top'] = int(grid.xpath('//top/text()')[0])
        config['step'] = int(grid.xpath('//step/text()')[0])
        config['nx'] = int(grid.xpath('//nX/text()')[0])
        config['ny'] = int(grid.xpath('//nY/text()')[0])    
    
    return config


def save_vectors_legacy(vectors, filename='output.txt', ssim_lim_lower=-1, assim_lim_upper=100):

    bad_vectors = [vector for vector in vectors if vector[8] <= ssim_lim_lower or vector[10] >= assim_lim_upper]
    good_vectors = [vector for vector in vectors if vector[8] > ssim_lim_lower and vector[10] < assim_lim_upper]

    with open(filename, mode='wt') as f:
        for lon0, lat0, lon1, lat1, x0, y0, x1, y1, ssim, v, assim in good_vectors:
            f.write(f'{lon0:.10f} {lat0:.10f} {lon1:.10f} {lat1:.10f} {x0} {y0} {x1} {y1} {ssim:.10f} {v:.10f} {assim:.10f}\n')
    
    with open(f'filtered_{filename}', mode='wt') as f:
        for lon0, lat0, lon1, lat1, x0, y0, x1, y1, ssim, v, assim in bad_vectors:
            f.write(f'{lon0:.10f} {lat0:.10f} {lon1:.10f} {lat1:.10f} {x0} {y0} {x1} {y1} {ssim:.10f} {v:.10f} {assim:.10f}\n')

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
        sa_right = sa_cx // 2 - rw_cx
        sa_top = - sa_cy // 2
        sa_bottom = sa_cy // 2 - rw_cy
        
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

    #print(f'Point_size = {point_size} / Total_size = {total_size} / Memory_limit = {memory_limit}')        
    batch_size = memory_limit // point_size
    return batch_size

def run_algorithm(file1=None, file2=None, config_name=None, output_name=None, disable_cuda=False, num_threads=4, element_size=8):

    
    b01, data1 = readproj(file1)
    b02, data2 = readproj(file2)

    file_config = read_config_legacy(config_name)

    mapper = projmapper.ProjMapper(
        lat=b01['b0_proj_common']['lat'], 
        lon=b01['b0_proj_common']['lon'], 
        lat_res=b01['b0_proj_common']['latRes']/3600, 
        lon_res=b01['b0_proj_common']['lonRes']/3600, 
        lat_size=b01['b0_proj_common']['latSize'], 
        lon_size=b01['b0_proj_common']['lonSize'], 
        pt=b01['b0_proj_common']['projType'] - 1
        )

    config = {
        'left': 150,
        'top': 150,
        'step': 5,
        'nx': 2,
        'ny': 2,
        'ssim_lim_lower': 0.8,
        'assim_lim_upper': 0.3,
        'search_radius_x': 20,
        'search_radius_y': 15,
        'window_radius_x': 10,
        'window_radius_y': 5,
        'element_size': 8,
        'memory_limit': 2,
        'device': 'auto',
        'sa_cx': 30, 
        'sa_cy': 30, 
        'rw_cx': 10,
        'rw_cy': 10,
        'legacy': True,
    }

    if file_config is not None:
        for key, value in file_config.items():
            config[key] = value

    if disable_cuda:
        config['device'] = 'cpu'
    config['element_size'] = element_size
    config['num_threads'] = num_threads
    config['batch_size'] = get_batch_size(config['memory_limit'], config)
    config['dt'] = calc_dt(b01, b02)
    config['pix_size_km'] = calc_pix_size_km(b01)
    config['mapper'] = mapper

    
    vectors_th_ref = vecthref.run_algorithm(data1, data2, config)

   # good_vectors, bad_vectors = filter_vectors(vectors, ssim_lim_lower=ssim_lim_lower, assim_lim_upper=assim_lim_upper)
    save_vectors_legacy(vectors_th_ref, filename=output_name, ssim_lim_lower=config['ssim_lim_lower'], assim_lim_upper=config['assim_lim_upper'])
    
    #save_vectors_for_glance(vectors, width=2, ssim_lim_lower=0.8, assim_lim_upper=0.1)

def main():
    argv = ['data/1.pro', 'data/2.pro', 'config.xml', 'output.txt']
    parser = argparse.ArgumentParser(description='Python implementation of Vecplotter algorithm.')
    parser.add_argument('input_file1', metavar='input_file1', nargs=1, help='pair of input images')
    parser.add_argument('input_file2', metavar='input_file1', nargs=1, help='pair of input images')
    parser.add_argument('config', metavar='config_file', nargs=1, help='config file')
    parser.add_argument('output', metavar='output_file', nargs=1, help='output file')
    
    args = parser.parse_args(argv)
    run_algorithm(*argv)

if __name__ == '__main__':
    main()