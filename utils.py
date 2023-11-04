
'''
utils.py

Some utility functions

'''

import scipy.ndimage as nd
import scipy.io as io
import matplotlib

import matplotlib.pyplot as plt
import skimage.measure as sk
from mpl_toolkits import mplot3d
import matplotlib.gridspec as gridspec
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import torch
import os
import pickle

import plotly.express as px
import pandas as pd

if torch.device('cuda' if torch.cuda.is_available() else 'cpu') != 'cpu':
    matplotlib.use('Agg')



def getVoxelFromMat(path, cube_len=64):
    if cube_len == 32:
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))

    else:
        # voxels = np.load(path) 
        # voxels = io.loadmat(path)['instance'] # 64x64x64
        # voxels = np.pad(voxels, (2, 2), 'constant', constant_values=(0, 0))
        # print (voxels.shape)
        voxels = io.loadmat(path)['instance'] # 30x30x30
        voxels = np.pad(voxels, (1, 1), 'constant', constant_values=(0, 0))
        voxels = nd.zoom(voxels, (2, 2, 2), mode='constant', order=0)
        # print ('here')
    # print (voxels.shape)
    return voxels


def getVFByMarchingCubes(voxels, threshold=0.5):
    v, f = sk.marching_cubes_classic(voxels, level=threshold)
    return v, f


def plotVoxelVisdom(voxels, visdom, title):
    v, f = getVFByMarchingCubes(voxels)
    visdom.mesh(X=v, Y=f, opts=dict(opacity=0.5, title=title))


def SavePloat_Voxels(voxels, path, iteration):
    voxels = voxels[:8].__ge__(0.5)
    fig = plt.figure(figsize=(32, 16))
    gs = gridspec.GridSpec(2, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(voxels):
        x, y, z = sample.nonzero()
        ax = plt.subplot(gs[i], projection='3d')
        ax.scatter(x, y, z, zdir='z', c='red')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_aspect('equal')
    # print (path + '/{}.png'.format(str(iteration).zfill(3)))
    plt.savefig(path + '/{}.png'.format(str(iteration).zfill(3)), bbox_inches='tight')
    plt.close()


class ShapeNetDataset(data.Dataset):

    def __init__(self, root, params):
        
        self.root = root
        self.listdir = os.listdir(self.root)
        # print (self.listdir)  
        # print (len(self.listdir)) # 10668

        data_size = len(self.listdir)
        # self.listdir = self.listdir[0:int(data_size*0.7)]
        self.listdir = self.listdir[0:int(data_size)]
        self.cube_len = params.cube_len
        
        print ('data_size =', len(self.listdir)) # train: 10668-1000=9668

    def __getitem__(self, index):
        with open(self.root + self.listdir[index], "rb") as f:
            volume = np.asarray(getVoxelFromMat(f, self.cube_len), dtype=np.float32)
            # print (volume.shape)
        return torch.FloatTensor(volume)

    def __len__(self):
        return len(self.listdir)


def generateZ(batch, params):

    if params.z_dis == "norm":
        Z = torch.Tensor(batch, params.z_dim).normal_(0, 0.33).to(params.device)
    elif params.z_dis == "uni":
        Z = torch.randn(batch, params.z_dim).to(params.device).to(params.device)
    else:
        print("z_dist is not normal or uniform")

    return Z


def save_train_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['train_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['train_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)


def save_val_log(writer, loss_D, loss_G, itr):
    scalar_info = {}
    for key, value in loss_G.items():
        scalar_info['val_loss_G/' + key] = value

    for key, value in loss_D.items():
        scalar_info['val_loss_D/' + key] = value

    for tag, value in scalar_info.items():
        writer.add_scalar(tag, value, itr)

def Plot_Save_3D_Tensor(tensor_data, model_saved_path, name, plot=False, save=True, threshold=0.5, opacity=0.7, color='red'):

    # Get the shape of the tensor
    shape = tensor_data.shape
    n_x, n_y, n_z = shape

    # use threshold to filter the value
    tensor_data[tensor_data < threshold] = 0

    # Create lists for x, y, and z coordinates
    x_coords = []
    y_coords = []
    z_coords = []

    for i in range(n_x):
        for j in range(n_y):
            for k in range(n_z):
                if tensor_data[i, j, k] > 0:
                    x_coords.append(i)
                    y_coords.append(j)
                    z_coords.append(k)

    # Create a DataFrame
    df = pd.DataFrame({
        'X': x_coords,
        'Y': y_coords,
        'Z': z_coords,
    })

    if plot:
        fig = px.scatter_3d(df, x='X', y='Y', z='Z', opacity=opacity, color_discrete_sequence=[color])
        # tight layout
        fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.show()  

    # Define the PLY file structure
    vertex = [(x, y, z) for x, y, z in zip(df['x'], df['y'], df['z'])]
    plydata = PlyData([PlyElement.describe(vertex, 'vertex')])

    # Save the PLY file
    plydata.write(model_saved_path + name + '_point_cloud.ply')

    return df
