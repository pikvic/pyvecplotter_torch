# Расчёт SSIM_LAB34
def lab34_ssim(x, y):
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


def calc_ssim_matrix_legacy(data1, data2, x0, y0, sa_cx, sa_cy, rw_cx, rw_cy):
    
    sa_left = x0 - sa_cx//2
    sa_right = x0 + sa_cx//2 - rw_cx
    sa_top = y0 - sa_cy//2
    sa_bottom = y0 + sa_cy//2 - rw_cy
    
    ssim_size_x = sa_right - sa_left
    ssim_size_y = sa_bottom - sa_top
    ssim_matrix = np.zeros((ssim_size_y, ssim_size_x))
    
    rw_left = x0 - rw_cx//2
    rw_right = x0 - rw_cx//2 + rw_cx
    rw_top = y0 - rw_cy//2
    rw_bottom = y0 - rw_cy//2 + rw_cy
    
    win1 = data1[rw_top : rw_bottom + 1, rw_left : rw_right + 1]
        
    for y in range(sa_top, sa_bottom):
        for x in range(sa_left, sa_right):
            win2 = data2[y: y+rw_cy +1, x: x+rw_cx +1]
            ssim_matrix[y-sa_top, x-sa_left] = lab34_ssim(win1, win2)
    
    return ssim_matrix



# Параметры алгоритма, как в конфиге
sa_cx = 11
sa_cy = 13
rw_cx = 3
rw_cy = 4
ssim_thr = 0.8
assim_thr = 0.05
v_thr = 1.9
C_pow = 1
E_pow = 1
S_pow = 1

left = 600
top = 500
step = 10
nx = 1
ny = 1

# Легаси алгоритм

vectors = []

for p in points:
    x0 = p[0]
    y0 = p[1]
    
    ssim12 = calc_ssim_matrix_legacy(data1, data2, x0, y0, sa_cx, sa_cy, rw_cx, rw_cy)
    
    ssim_max = ssim12.max()
       
    sa_left = x0 - sa_cx//2
    sa_right = x0 + sa_cx//2 - rw_cx
    sa_top = y0 - sa_cy//2
    sa_bottom = y0 + sa_cy//2 - rw_cy    
    
    i, j = np.unravel_index(ssim12.argmax(), ssim12.shape) 
    
    y1 = sa_top + i + rw_cy//2
    x1 = sa_left + j + rw_cx//2
    
    ssim11 = calc_ssim_matrix_legacy(data1, data1, x0, y0, sa_cx, sa_cy, rw_cx, rw_cy)
    ssim22 = calc_ssim_matrix_legacy(data2, data2, x1, y1, sa_cx, sa_cy, rw_cx, rw_cy)
    
    a = np.argwhere(ssim11 == 1.0)
    b = np.argwhere(ssim11 > ssim_max)
    r1 = np.sqrt(np.sum((a-b)**2, axis=1))
    r1 = np.max(r1)
    
    
    a = np.argwhere(ssim11 == 1.0)
    b = np.argwhere(ssim11 > ssim_max)
    r2 = np.sqrt(np.sum((a-b)**2, axis=1))
    r2 = np.max(r2)

    r_max = max(r1, r2)
    
    # assim = pix_size_km * r_max * 1000 / d_t
    # v = pix_size_km * np.sqrt(np.power(x1 - x0, 2) + np.power(y1-y0, 2))*1000/d_t
    
    # lon0 = mapper.lon(column=x0)
    # lat0 = mapper.lat(scan=y0)
    # lon1 = mapper.lon(column=x1)
    # lat1 = mapper.lat(scan=y1)
    
    vectors.append([x0, y0, x1, y1, ssim_max, r_max])
   
    
print(vectors)


# Чтение конфига
from lxml import etree

path_to_config = r'D:\_PROJECTS\_LAB34_2018\vecplotter_pikvic\Debug\config.xml'


with open(path_to_config, 'rt') as f:
    xml = f.read()

tree = etree.fromstring(xml)

sa_cx = int(tree.xpath('/config/sa_cx/text()')[0])
sa_cy = int(tree.xpath('/config/sa_cy/text()')[0])

iterations = tree.xpath('//iteration')
for iteration in iterations:
    rw_cx = int(iteration.xpath('//rw_cx/text()')[0])
    rw_cy = int(iteration.xpath('//rw_cy/text()')[0])
    ssim_thr = float(iteration.xpath('//SSIM_thr/text()')[0])
    assim_thr = float(iteration.xpath('//aSSIM_thr/text()')[0])
    v_thr = float(iteration.xpath('//v_thr/text()')[0])
    C_pow = int(iteration.xpath('//C/text()')[0])
    E_pow = int(iteration.xpath('//E/text()')[0])
    S_pow = int(iteration.xpath('//S/text()')[0])

    
grids = tree.xpath('//grids/regular5')
for grid in grids:
    left = int(grid.xpath('//left/text()')[0])
    top = int(grid.xpath('//top/text()')[0])
    step = int(grid.xpath('//step/text()')[0])
    nx = int(grid.xpath('//nX/text()')[0])
    ny = int(grid.xpath('//nY/text()')[0])