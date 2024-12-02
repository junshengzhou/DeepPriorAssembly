import numpy as np
import open3d as o3d
import cv2
import os
import torch
from reg_7dof import register_mesh
import sys

def depth_image_to_point_cloud(depth, scale, K):
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    K[0, 2], K[1, 2] = depth.shape[1] / 2, depth.shape[0] / 2

    Z = depth.astype(float) / scale

    X = (u  + 0.5 - K[0, 2]) * Z / K[0, 0]
    Y = (v  + 0.5 - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = -Y[valid]
    Z = -Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))

    points = position[0:3, :]

    return points

def projection_pc_img(pc_np, K):

    pc_np = pc_np.transpose(1,0)

    pc_np_front = pc_np

    pc_pixels = np.dot(K, pc_np_front) 
    pc_pixels = pc_pixels / pc_pixels[2:, :] 

    return pc_pixels

def projection_pc_img_gpu(pc_np, K):

    pc_np = pc_np.transpose(1,0)

    pc_np_front = pc_np 

    pc_pixels = torch.dot(K, pc_np_front)  
    pc_pixels = pc_pixels / pc_pixels[2:, :]  

    return pc_pixels


def pc_normalize(pc):

    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m

    return pc

def gen_instant_pcs(depth, masks, intri):

    pcs_all = []
    for mask in masks:
        mask = mask.astype(np.bool)
        resized_mask = mask
        mask_inverse = 1 - resized_mask
        masked_region_white = -1 * mask_inverse

        depth_seg = resized_mask * depth + masked_region_white

        pcs = depth_image_to_point_cloud(depth_seg, 1000, intri)

        pcs_all.append(pcs)
    
    return pcs_all

def gen_instant_pcs_pano(geometry_path, num):

    pcs_all = []

    for i in range(num):

        file_path = os.path.join(geometry_path, 'instance_%d.xyz'%i)
        
        pcs = np.loadtxt(file_path)

        pcs_all.append(pcs)
    
    return pcs_all

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--image_id', type=str, default='')

parser.add_argument('--output_dir',type=str, default='')

parser.add_argument('--geometry_dir',type=str, default='')

parser.add_argument('--mask_dir',type=str, default='')

parser.add_argument('--object_dir',type=str, default='')



opt = parser.parse_args()


focals = np.load(os.path.join(opt.geometry_dir, opt.image_id, 'focal.npy')) 

intri = np.identity(3)
intri[0][0], intri[1][1] = focals[0][0], focals[1][0]

opt.output_path = os.path.join(opt.output_dir, opt.image_id)
os.makedirs(opt.output_path, exist_ok=True)

exp_name = opt.output_path

mask_path = os.path.join(opt.mask_dir, opt.image_id, 'mask_all.npy')

masks = np.load(mask_path)

instant_pcs = gen_instant_pcs_pano(os.path.join(opt.geometry_dir, opt.image_id), len(masks))

print('The scene has %d instants. ' % len(masks))


merged_mesh = o3d.geometry.TriangleMesh()

for i in range(len(instant_pcs)):

    target_points = instant_pcs[i]

    mask = masks[i]

    min_metric = 100

    for j in range(3):
        for k in range(2):
            folder_path = os.path.join(opt.object_dir, opt.image_id)
            items = os.listdir(folder_path)
            mesh_folders = [item for item in items if os.path.isdir(os.path.join(folder_path, item)) and item.startswith('seg_%d'%i)]

            mesh_path = folder_path + '/' +  mesh_folders[0] + "/filter_0000%d/mesh_%d.obj" % (j, k)
            print(mesh_path)
            mesh, loss_3d, loss_2d = register_mesh(i, target_points, mesh_path, intri, os.path.join(exp_name, str(i)))
            if loss_3d + loss_2d / 200 < min_metric:
                min_metric = loss_3d + loss_2d / 200
                mesh_best = mesh

    print(min_metric)
    
    merged_mesh += mesh_best
    o3d.io.write_triangle_mesh(os.path.join(exp_name, "best_mesh_%d.obj"%i), mesh_best)
    
o3d.io.write_triangle_mesh(os.path.join(exp_name, "merged_mesh.obj"), merged_mesh)
