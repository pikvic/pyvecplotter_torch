import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from numpy.lib.stride_tricks import as_strided

# TODO refactor for x and y radiuses

class PairsDataset(Dataset):
   
    def __init__(self, data1, data2, points, search_radius, window_radius):
        
        self.search_radius = search_radius
        self.window_radius = window_radius
        self.ssim_radius = self.search_radius - self.window_radius
        
        self.search_size = search_radius * 2 + 1
        self.window_size = window_radius * 2 + 1
        self.ssim_size = self.ssim_radius * 2 + 1

        self.ssim_numel = self.ssim_size * self.ssim_size
        self.data1 = data1
        self.data2 = data2
        self.points = points
        
    def __len__(self):
        return len(self.points)

    def __getitem__(self, idx):

        x0 = self.points[idx][0]
        y0 = self.points[idx][1]
       
        x_beg, x_end = x0 - self.search_radius, x0 + self.search_radius + 1
        y_beg, y_end = y0 - self.search_radius, y0 + self.search_radius + 1 
       
        area1 = self.data1[y_beg:y_end, x_beg:x_end]
        area2 = self.data2[y_beg:y_end, x_beg:x_end]
        
        beg, end = self.search_radius - self.window_radius, self.search_radius + self.window_radius + 1
        win1 = area1[beg:end, beg:end]

        win2 = as_strided(area2, (self.ssim_size, self.ssim_size, self.window_size, self.window_size), area2.strides + area2.strides)

        return (
            torch.from_numpy(win1.reshape(1, self.window_size * self.window_size)), 
            torch.from_numpy(win2.reshape(self.ssim_size * self.ssim_size, self.window_size * self.window_size))
        )

def ssim(x, y):

    x_mean = x.mean(dim=2, keepdim=True)
    y_mean = y.mean(dim=2, keepdim=True)
    
    x_diff = x - x_mean
    y_diff = y - y_mean
    
    x_std = x.std(dim=2)
    y_std = y.std(dim=2)
     
    tC = torch.sum(x_diff*y_diff, dim=2) / torch.sqrt(torch.sum(x_diff*x_diff, dim=2) * torch.sum(y_diff*y_diff, dim=2))
    tE = 1 - torch.sum(torch.abs(x_diff - y_diff), dim=2) / (torch.sum(torch.abs(x_diff), dim=2) + torch.sum(torch.abs(y_diff), dim=2))
    tS = 2 * x_std*y_std / (x_std*x_std + y_std*y_std)
  
    return tC * tE * tS


# TODO dataset and ssim done. refactor run_algorithm

def ssim_matrix(data1, data2, points, search_radius, window_radius):
    batch_size = 200
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = PairsDataset(data1, data2, points, search_radius, window_radius)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        
        ssim_vector = torch.zeros((len(dataset), dataset.ssim_size*dataset.ssim_size), device=device, dtype=torch.float64)
        
        for i_batch, (win1, win2) in enumerate(dataloader):
                    
            win1 = win1.to(device)
            win2 = win2.to(device)    
            
            ssim_vector_batched = ssim(win1, win2)

            beg, end = i_batch*batch_size, i_batch*batch_size + batch_size

            ssim_vector[beg:end, :] = ssim_vector_batched
        
    return  ssim_vector.cpu().numpy().reshape(len(dataset), dataset.ssim_size, dataset.ssim_size)


def process_points(data1, data2, points, search_radius, window_radius):

    ssim12 = ssim_matrix(data1, data2, points, search_radius, window_radius)
    ssim11 = ssim_matrix(data1, data1, points, search_radius, window_radius)

    new_points = []

    for idx, (x0, y0) in enumerate(points):
        ssim_max = np.max(ssim12, axis=(1,2))
        i, j = np.unravel_index(ssim12[idx, :].argmax(), ssim12[idx, :].shape)
        y1 = y0 - search_radius + window_radius + i
        x1 = x0 - search_radius + window_radius + j
        
        new_points.append((x1, y1))

    ssim22 = ssim_matrix(data2, data2, new_points, search_radius, window_radius)

    vectors = []
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
        vector = (x0, y0, x1, y1, ssim_max[idx], r_max)
        vectors.append(vector)

    return vectors


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
    
    print(f'Running algorigthm with parameters:\n')
    print(f'search_radius_x={search_radius_x} search_radius_y={search_radius_y} window_radius_x={window_radius_x} window_radius_y={window_radius_y}')
    print(f'left={left} top={top} step={step} nx={nx} ny={ny}')

    vectors = process_points(data1, data2, points, search_radius_x, window_radius_x)

    for vector in vectors:
        print(vector)
    
