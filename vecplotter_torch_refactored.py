import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import as_strided

# TODO make an option for auto or manual setting of memory_limit - use cudainfo and psutils

class PairsDataset(Dataset):
   
    def __init__(self, data1, data2, points, search_radius_x, search_radius_y, window_radius_x, window_radius_y, dtype=torch.float64, legacy=False):
        self.dtype = dtype
        self.legacy = legacy
        self.search_radius_x = search_radius_x
        self.search_radius_y = search_radius_y
        self.window_radius_x = window_radius_x
        self.window_radius_y = window_radius_y
        self.ssim_radius_x = search_radius_x - window_radius_x
        self.ssim_radius_y = search_radius_y - window_radius_y
        
        self.search_size_x = search_radius_x * 2 + 1
        self.search_size_y = search_radius_y * 2 + 1
        self.window_size_x = window_radius_x * 2 + 1
        self.window_size_y = window_radius_y * 2 + 1
        self.ssim_size_x = self.ssim_radius_x * 2 + 1
        self.ssim_size_y = self.ssim_radius_y * 2 + 1

        self.ssim_numel = self.ssim_size_x * self.ssim_size_y
        self.window_numel = self.window_size_x * self.window_size_y

        self.data1 = data1
        self.data2 = data2
        self.points = points
        self.width = data1.shape[1]
        self.height = data1.shape[0]

        if self.legacy:

            sa_left = - self.search_radius_x // 2
            sa_right = self.search_radius_x // 2 - self.window_radius_x
            sa_top = - self.search_radius_y // 2
            sa_bottom = self.search_radius_y // 2 - self.window_radius_y
            
            ssim_size_x = sa_right - sa_left
            ssim_size_y = sa_bottom - sa_top
            self.ssim_size_x = ssim_size_x
            self.ssim_size_y = ssim_size_y
            self.ssim_numel = ssim_size_x * ssim_size_y


    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):

        x0 = self.points[idx][0]
        y0 = self.points[idx][1]
     
        x_beg, x_end = x0 - self.search_radius_x, x0 + self.search_radius_x + 1
        y_beg, y_end = y0 - self.search_radius_y, y0 + self.search_radius_y + 1 
       
        # TODO check points and windows are not out of borders if so fill with zeros

        area1 = self.data1[y_beg:y_end, x_beg:x_end]
        area2 = self.data2[y_beg:y_end, x_beg:x_end]

        x_beg, x_end = self.search_radius_x - self.window_radius_x, self.search_radius_x + self.window_radius_x + 1
        y_beg, y_end = self.search_radius_y - self.window_radius_y, self.search_radius_y + self.window_radius_y + 1
        
        win1 = area1[y_beg:y_end, x_beg:x_end]

        win2 = as_strided(area2, (self.ssim_size_y, self.ssim_size_x, self.window_size_y, self.window_size_x), area2.strides + area2.strides)

        if self.legacy:

            sa_left = x0 - self.search_radius_x // 2
            sa_right = x0 + self.search_radius_x // 2 - self.window_radius_x
            sa_top = y0 - self.search_radius_y // 2
            sa_bottom = y0 + self.search_radius_y // 2 - self.window_radius_y
            
            ssim_size_x = sa_right - sa_left
            ssim_size_y = sa_bottom - sa_top
            self.ssim_numel = ssim_size_x * ssim_size_y


            rw_left = x0 - self.window_radius_x // 2
            rw_right = x0 - self.window_radius_x // 2 + self.window_radius_x
            rw_top = y0 - self.window_radius_y // 2
            rw_bottom = y0 - self.window_radius_x // 2 + self.window_radius_x
            
            window_size_x = rw_right - rw_left + 1
            window_size_y = rw_bottom - rw_top + 1

            win1 = self.data1[rw_top : rw_bottom + 1, rw_left : rw_right + 1]
            win2 = np.zeros((ssim_size_y, ssim_size_x, window_size_y, window_size_x))
            
            for y in range(sa_top, sa_bottom):
                for x in range(sa_left, sa_right):
                    win2[y - sa_top, x - sa_left, :, :] = self.data2[y: y + self.window_radius_y + 1, x: x + self.window_radius_x + 1]
            return (
                torch.from_numpy(win1.reshape(1, window_size_x*window_size_y)).type(self.dtype), 
                torch.from_numpy(win2.reshape(ssim_size_x*ssim_size_y, window_size_x*window_size_y)).type(self.dtype)    
            )

        return (
            torch.from_numpy(win1.reshape(1, self.window_numel)).type(self.dtype), 
            torch.from_numpy(win2.reshape(self.ssim_numel, self.window_numel)).type(self.dtype)
        )


def _ssim(x, y):
    x_mean = x.mean(dim=2, keepdim=True)
    y_mean = y.mean(dim=2, keepdim=True)
    x_std = x.std(dim=2)
    y_std = y.std(dim=2)
    x.sub_(x_mean)
    y.sub_(y_mean)
    sum12 = torch.sum(x * y, dim=2)
    sum11 = torch.sum(x * x, dim=2)
    sum22 = torch.sum(y * y, dim=2)
    tC = sum12 / torch.sqrt(sum11 * sum22)
    sum12 = torch.sum(torch.abs(x - y), dim=2)
    sum11 = torch.sum(x.abs_(), dim=2)
    sum22 = torch.sum(y.abs_(), dim=2)
    tE = 1 - sum12 / (sum11 + sum22)
    tS = 2 * x_std*y_std / (x_std*x_std + y_std*y_std)
    
    # tC = torch.sum(x_diff*y_diff, dim=2) / torch.sqrt(torch.sum(x_diff*x_diff, dim=2) * torch.sum(y_diff*y_diff, dim=2))
    # tE = 1 - torch.sum(torch.abs(x_diff - y_diff), dim=2) / (torch.sum(torch.abs(x_diff), dim=2) + torch.sum(torch.abs(y_diff), dim=2))
    # print(f'CUDA info 3: {torch.cuda.memory_allocated() / 1024 / 1024} / {torch.cuda.max_memory_allocated()/ 1024 / 1024}')
   
    return tC * tE * tS


def _ssim_matrix(data1, data2, points, config):
    
    search_radius_x = config['search_radius_x']
    search_radius_y = config['search_radius_y']
    window_radius_x = config['window_radius_x']
    window_radius_y = config['window_radius_y']
    batch_size = config['batch_size']
    device = config['device']
    element_size = config['element_size']
    legacy = config['legacy']
    sa_cx = config['sa_cx']
    sa_cy = config['sa_cy']
    rw_cx = config['rw_cx']
    rw_cy = config['rw_cy']

    if element_size == 2:
        dtype = torch.float16
    elif element_size == 4:
        dtype = torch.float32
    elif element_size == 8:
        dtype = torch.float64
    else:
        dtype = torch.float64

    if device == 'auto':
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    if not legacy:
        dataset = PairsDataset(data1, data2, points, search_radius_x, search_radius_y, window_radius_x, window_radius_y, dtype=dtype, legacy=legacy)

    else:
        dataset = PairsDataset(data1, data2, points, sa_cx, sa_cy, rw_cx, rw_cy, dtype=dtype, legacy=legacy)

    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    with torch.no_grad():
        
        ssim_vector = torch.zeros((len(dataset), dataset.ssim_numel), device=device, dtype=dtype)
       
        for i_batch, (win1, win2) in enumerate(dataloader):

            win1 = win1.to(device)
            win2 = win2.to(device)    
            
            ssim_vector_batched = _ssim(win1, win2)

            beg, end = i_batch*batch_size, i_batch*batch_size + batch_size

            ssim_vector[beg:end, :] = ssim_vector_batched

    return  ssim_vector.cpu().numpy().reshape(len(dataset), dataset.ssim_size_y, dataset.ssim_size_x)


def run_algorithm(data1, data2, config):

    #torch.set_num_threads(1)

    left = config['left']
    top = config['top']
    step = config['step']
    nx = config['nx']
    ny = config['ny']
    search_radius_x = config['search_radius_x']
    search_radius_y = config['search_radius_y']
    window_radius_x = config['window_radius_x']
    window_radius_y = config['window_radius_y']
    sa_cx = config['sa_cx']
    sa_cy = config['sa_cy']
    rw_cx = config['rw_cx']
    rw_cy = config['rw_cy']
    legacy = config['legacy']
    batch_size = config['batch_size']
    dt = config['dt']
    pix_size_km = config['pix_size_km']
    mapper = config['mapper']

    points = [(x, y) for x in range(left, left + step*nx, step) for y in range(top, top + step*ny, step)]
    
    # if not legacy:
    #     print(f'Running algorigthm with parameters:')
    #     print(f'search_radius_x={search_radius_x} search_radius_y={search_radius_y} window_radius_x={window_radius_x} window_radius_y={window_radius_y}')
    #     print(f'left={left} top={top} step={step} nx={nx} ny={ny}')
    # else:
    #     print(f'Running algorighm with parameters:\n')
    #     print(f'sa_cx={sa_cx} sa_cy={sa_cy} rw_cx={rw_cx} rw_cy={rw_cy}')
    #     print(f'left={left} top={top} step={step} nx={nx} ny={ny}')

    # print('Ssim 1 to 2...')
    ssim12 = _ssim_matrix(data1, data2, points, config)
    # print('Ssim 1 to 1...')
    ssim11 = _ssim_matrix(data1, data1, points, config)

    new_points = []

    # TODO Refactor this for vector not matrix
    for idx, (x0, y0) in enumerate(points):
        ssim_max = np.max(ssim12, axis=(1,2))
        i, j = np.unravel_index(ssim12[idx, :].argmax(), ssim12[idx, :].shape)
        if not legacy:
            y1 = y0 - search_radius_x + window_radius_x + i
            x1 = x0 - search_radius_x + window_radius_x + j
        else:
            sa_left = x0 - sa_cx//2
            sa_top = y0 - sa_cy//2
            y1 = sa_top + i + rw_cy//2
            x1 = sa_left + j + rw_cx//2

        new_points.append((x1, y1))
    
    # print('Ssim 1 to 2...')
    ssim22 = _ssim_matrix(data2, data2, new_points, config)

    vectors = []

    # TODO Refactor this for vector not matrix
    for idx, (p0, p1) in enumerate(zip(points, new_points)):
        id1_max = np.argwhere(ssim11[idx, :] == np.max(ssim11[idx, :]))
        id2_max = np.argwhere(ssim22[idx, :] == np.max(ssim22[idx, :]))
        id1 = np.argwhere(ssim11[idx, :] > ssim_max[idx])
        id2 = np.argwhere(ssim22[idx, :] > ssim_max[idx])
        r1 = np.max(np.sqrt(np.einsum('ij,ij->i', id1 - id1_max, id1-id1_max)))
        r2 = np.max(np.sqrt(np.einsum('ij,ij->i', id2 - id2_max, id2-id2_max)))
 
        r_max = max(r1, r2)
        x0, y0 = p0
        x1, y1 = p1
        assim = pix_size_km * r_max * 1000 / dt
        v = pix_size_km * np.sqrt(np.power(x1 - x0, 2) + np.power(y1-y0, 2))*1000/dt
        lon0 = mapper.lon(column=x0)
        lat0 = mapper.lat(scan=y0)
        lon1 = mapper.lon(column=x1)
        lat1 = mapper.lat(scan=y1)
        vector = (lon0, lat0, lon1, lat1, x0, y0, x1, y1, ssim_max[idx], v, assim)
        vectors.append(vector)

    return vectors
    
