import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# TODO flatten windows (refactor Dataset)
# TODO allocate all data on device
# TODO calculate memory consumption
# TODO make bigger batches
# TODO refactor functions
# TODO make zeros ssim_matrix vector and write batched results there

class PairsDataset(Dataset):
   
    def __init__(self, area1, area2, search_radius, window_radius):
        
        self.search_radius = search_radius
        self.window_radius = window_radius
        self.ssim_radius = self.search_radius - self.window_radius
        
        self.search_size = search_radius * 2 + 1
        self.window_size = window_radius * 2 + 1
        self.ssim_size = self.ssim_radius * 2 + 1

        self.ssim_count = self.ssim_size * self.ssim_size
        self.area1 = area1
        self.area2 = area2
        
    def __len__(self):
        return self.ssim_count

    def __getitem__(self, idx):
        i = idx // self.ssim_size
        j = idx % self.ssim_size
        
        beg, end = self.search_radius - self.window_radius, self.search_radius + self.window_radius + 1
        win1 = self.area1[beg:end, beg:end]
        
        win2 = self.area2[i:i+self.window_size, j:j+self.window_size]

        return (torch.from_numpy(win1.reshape(1, self.window_size, self.window_size)), torch.from_numpy(win2.reshape(1, self.window_size, self.window_size)))


def ssim(x, y):

    x_mean = x.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    y_mean = y.mean(dim=3, keepdim=True).mean(dim=2, keepdim=True)
    
    x_diff = x - x_mean
    y_diff = y - y_mean
    
    x_std = x.flatten(-2, -1).std(dim=2)
    y_std = y.flatten(-2, -1).std(dim=2)
     
    tC = torch.sum(x_diff*y_diff, dim=3).sum(dim=2) / torch.sqrt(torch.sum(x_diff*x_diff, dim=3).sum(dim=2)*torch.sum(y_diff*y_diff, dim=3).sum(dim=2))
    tE = 1 - torch.sum(torch.abs(x_diff - y_diff), dim=3).sum(dim=2) / (torch.sum(torch.abs(x_diff), dim=3).sum(dim=2) + torch.sum(torch.abs(y_diff), dim=3).sum(dim=2))
    tS = 2 * x_std*y_std / (x_std*x_std + y_std*y_std)
    
    return (tC * tE * tS).squeeze(-1)

def ssim_matrix(area1, area2, search_radius, window_radius):
    batch_size = 2
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    dataset = PairsDataset(area1, area2, search_radius, window_radius)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():

        ssim_batches = []
        
        for i_batch, (win1, win2) in enumerate(dataloader):
                    
            win1 = win1.to(device)
            win2 = win2.to(device)    
            
            ssim_batches.append(ssim(win1, win2))
        
        ssim_vector = torch.cat(ssim_batches, 0).cpu()
        
    return ssim_vector.reshape(search_radius*2 - window_radius*2+1, search_radius*2 - window_radius*2+1).numpy()


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
    
    print(f'Running algorigthm with parameters:\n')
    print(f'search_radius_x={search_radius_x} search_radius_y={search_radius_y} window_radius_x={window_radius_x} window_radius_y={window_radius_y}')
    print(f'left={left} top={top} step={step} nx={nx} ny={ny}')

    for point_num, (x0, y0) in enumerate(points):
        print(f'Processing point  {point_num+1}/{len(points)} : ({x0}, {y0})...')
        vector = process_point(data1, data2, x0, y0, search_radius_x, window_radius_x)
        vectors.append(vector)

    for vector in vectors:
        print(vector)
    
