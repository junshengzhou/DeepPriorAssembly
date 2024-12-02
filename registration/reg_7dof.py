import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

def save_pcs(source, target, pred, name):
    cloud_a = o3d.geometry.PointCloud()
    cloud_b = o3d.geometry.PointCloud()
    cloud_c = o3d.geometry.PointCloud()

    cloud_a.points = o3d.utility.Vector3dVector(source)
    cloud_b.points = o3d.utility.Vector3dVector(target)
    cloud_c.points = o3d.utility.Vector3dVector(pred)

    color_a = [1, 0, 0]  
    color_b = [0, 1, 0]  
    color_c = [0, 0, 1]  

    cloud_a.paint_uniform_color(color_a)
    cloud_b.paint_uniform_color(color_b)
    cloud_c.paint_uniform_color(color_c)

    combined_cloud = o3d.geometry.PointCloud()
    combined_cloud += cloud_a
    combined_cloud += cloud_b
    combined_cloud += cloud_c

    o3d.io.write_point_cloud(name, combined_cloud)

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc_new = pc - centroid
    m = np.max(np.sqrt(np.sum(pc_new ** 2, axis=1)))
    pc_new = pc_new / m

    return pc_new, centroid, m



class PointCloudRegistrationModel(nn.Module):
    def __init__(self):
        super(PointCloudRegistrationModel, self).__init__()
        self.rotation_x = nn.Parameter(torch.zeros(1, device="cuda"), requires_grad=True)
        self.rotation_y = nn.Parameter(torch.zeros(1, device="cuda"), requires_grad=True)
        self.rotation_z = nn.Parameter(torch.zeros(1, device="cuda"), requires_grad=True)
        self.translation = nn.Parameter(torch.zeros(3, device="cuda"), requires_grad=True)
        self.scale = nn.Parameter(torch.ones(1, device="cuda"), requires_grad=True)

    def forward(self, source_points):
        cos_theta_x = torch.cos(self.rotation_x)
        sin_theta_x = torch.sin(self.rotation_x)
        rotation_matrix_x = torch.zeros(3, 3, device="cuda")
        rotation_matrix_x[0, 0] = 1
        rotation_matrix_x[1, 1] = cos_theta_x
        rotation_matrix_x[1, 2] = -sin_theta_x
        rotation_matrix_x[2, 1] = sin_theta_x
        rotation_matrix_x[2, 2] = cos_theta_x

        cos_theta_y = torch.cos(self.rotation_y)
        sin_theta_y = torch.sin(self.rotation_y)
        rotation_matrix_y = torch.zeros(3, 3, device="cuda")
        rotation_matrix_y[0, 0] = cos_theta_y
        rotation_matrix_y[0, 2] = sin_theta_y
        rotation_matrix_y[1, 1] = 1
        rotation_matrix_y[2, 0] = -sin_theta_y
        rotation_matrix_y[2, 2] = cos_theta_y

        cos_theta_z = torch.cos(self.rotation_z)
        sin_theta_z = torch.sin(self.rotation_z)
        rotation_matrix_z = torch.zeros(3, 3, device="cuda")
        rotation_matrix_z[0, 0] = cos_theta_z
        rotation_matrix_z[0, 1] = -sin_theta_z
        rotation_matrix_z[1, 0] = sin_theta_z
        rotation_matrix_z[1, 1] = cos_theta_z
        rotation_matrix_z[2, 2] = 1

        rotation_matrix = torch.matmul(rotation_matrix_z, torch.matmul(rotation_matrix_y, rotation_matrix_x))

        transformed_points = torch.matmul(source_points, rotation_matrix) + self.translation
        transformed_points *= self.scale[0]

        return transformed_points
    
    def get_transform(self):

        cos_theta_x = torch.cos(self.rotation_x)
        sin_theta_x = torch.sin(self.rotation_x)
        rotation_matrix_x = torch.zeros(3, 3, device="cuda")
        rotation_matrix_x[0, 0] = 1
        rotation_matrix_x[1, 1] = cos_theta_x
        rotation_matrix_x[1, 2] = -sin_theta_x
        rotation_matrix_x[2, 1] = sin_theta_x
        rotation_matrix_x[2, 2] = cos_theta_x

        cos_theta_y = torch.cos(self.rotation_y)
        sin_theta_y = torch.sin(self.rotation_y)
        rotation_matrix_y = torch.zeros(3, 3, device="cuda")
        rotation_matrix_y[0, 0] = cos_theta_y
        rotation_matrix_y[0, 2] = sin_theta_y
        rotation_matrix_y[1, 1] = 1
        rotation_matrix_y[2, 0] = -sin_theta_y
        rotation_matrix_y[2, 2] = cos_theta_y

        cos_theta_z = torch.cos(self.rotation_z)
        sin_theta_z = torch.sin(self.rotation_z)
        rotation_matrix_z = torch.zeros(3, 3, device="cuda")
        rotation_matrix_z[0, 0] = cos_theta_z
        rotation_matrix_z[0, 1] = -sin_theta_z
        rotation_matrix_z[1, 0] = sin_theta_z
        rotation_matrix_z[1, 1] = cos_theta_z
        rotation_matrix_z[2, 2] = 1

        rotation_matrix = torch.matmul(rotation_matrix_z, torch.matmul(rotation_matrix_y, rotation_matrix_x))

        return rotation_matrix, self.translation, self.scale[0]


def projection_pc_img_gpu(pc_np, K):
    pc_np_front = pc_np.transpose(1,0) 

    pc_pixels = torch.matmul(K, pc_np_front) 
    pc_pixels = pc_pixels / pc_pixels[2:, :] 

    return pc_pixels


def chamfer_distance_loss(source_points, target_points):
    source_to_target_dist = torch.cdist(source_points, target_points)
    target_to_source_dist = torch.cdist(target_points, source_points)
    loss = torch.mean(torch.min(source_to_target_dist, dim=1)[0]) + torch.mean(torch.min(target_to_source_dist, dim=1)[0])
    return loss

def rotation_matrix_loss(rotation_matrix):
    ortho_penalty = torch.norm(torch.eye(3).cuda() - torch.matmul(rotation_matrix.t(), rotation_matrix))
    
    norm_penalty = torch.norm(rotation_matrix, dim=0) - 1
    norm_penalty = torch.mean(norm_penalty**2)
    
    return ortho_penalty + norm_penalty

def rotate_x_matrix(theta):
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])
    return rotation_matrix

def rotate_y_matrix(theta):
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])
    return rotation_matrix

def rotate_z_matrix(theta):
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return rotation_matrix

def random_rotation_matrix():
    theta_x = np.random.uniform(0, 2 * np.pi)
    theta_y = np.random.uniform(0, 2 * np.pi)
    theta_z = np.random.uniform(0, 2 * np.pi)

    # Create rotation matrices
    rotation_matrix_x = rotate_x_matrix(theta_x)
    rotation_matrix_y = rotate_y_matrix(theta_y)
    rotation_matrix_z = rotate_z_matrix(theta_z)

    # Combine the rotation matrices
    rotation_matrix = np.dot(rotation_matrix_z, np.dot(rotation_matrix_y, rotation_matrix_x))

    return rotation_matrix

def apply_rotation_matrix(rotation_matrix, point):
    rotated_point = np.dot(rotation_matrix, point)
    return rotated_point

def random_rotation(point):
    rotation_matrix = random_rotation_matrix()
    rotated_point = apply_rotation_matrix(rotation_matrix, point)
    return rotated_point


def transform_mesh_o3d(mesh, rotation_matrix=None, translation_vector=None, scale_factor=None, direct_metrix=None):
    if direct_metrix is not None:
        mesh.transform(direct_metrix)
        return mesh
    
    transformation = np.eye(4) 
    if rotation_matrix is not None:
        transformation[:3, :3] = rotation_matrix  
    if translation_vector is not None:
        transformation[:3, 3] = translation_vector  
    if scale_factor is not None:
        transformation[0, 0] *= scale_factor  
        transformation[1, 1] *= scale_factor  
        transformation[2, 2] *= scale_factor  

    mesh.transform(transformation)
    return mesh

def remove_outliers(points, threshold=2.0):
    mean = np.mean(points, axis=0)
    std = np.std(points, axis=0)
    is_outlier = np.any(np.abs((points - mean) / std) > threshold, axis=1)
    cleaned_points = points[~is_outlier]
    return cleaned_points

def register_mesh(ind, target_points, gen_mesh_path, intri, exp_name):
    # ------------------------3d transform----------------------------
    min_cd = 1000
    all_cd = []

    target_points = remove_outliers(target_points, threshold=10)


    _, target_center, target_scale = pc_normalize(target_points)

    ran_ind = np.random.choice(target_points.shape[0], 5000)
    target_points = target_points[ran_ind]


    target_points = torch.from_numpy(target_points).float().cuda()

    mesh = o3d.io.read_triangle_mesh(gen_mesh_path)
    mesh_pointcloud = mesh.sample_points_poisson_disk(number_of_points=5000)
    mesh_pointcloud = np.asarray(mesh_pointcloud.points)
    mesh_pointcloud, source_center, source_scale = pc_normalize(mesh_pointcloud)

    mesh.translate(-source_center)
    mesh.scale(1/source_scale, np.array([0.0, 0.0, 0.0]))

    intri = torch.FloatTensor(intri).cuda()
    
    for rate in range(2):
        print('Processing instant %d, random time %d ' % (ind, rate))

        random_metrix = random_rotation_matrix()

        mesh_pointcloud_rotate = apply_rotation_matrix(mesh_pointcloud, random_metrix)

        mesh_pointcloud_rotate = mesh_pointcloud_rotate * target_scale + target_center


        source_points = torch.from_numpy(mesh_pointcloud_rotate).float().cuda()

        # ------------------------2d matching----------------------------


        target_points_2d = projection_pc_img_gpu(target_points, intri)
        source_points_2d = projection_pc_img_gpu(source_points, intri)

        # ---------------------------------------------------------------


        model = PointCloudRegistrationModel().cuda()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        num_epochs = 1000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            
            transformed_points = model(source_points)
            
            transformed_points_2d = projection_pc_img_gpu(transformed_points, intri)

            loss_3d = chamfer_distance_loss(transformed_points, target_points)

            loss_2dmatching = chamfer_distance_loss(transformed_points_2d.transpose(1,0), target_points_2d.transpose(1,0)) 

            if epoch < 600:
                loss = loss_3d
            else:
                loss = loss_3d+loss_2dmatching

            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) == 600 or (epoch + 1) == 1000:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss_3d: {loss_3d.item():.4f}, Loss_2d: {loss_2dmatching.item():.4f}')

            if epoch == num_epochs-1:
                os.makedirs('%s'%exp_name, exist_ok=True)
                
                save_pcs(source_points.detach().cpu().numpy(), target_points.detach().cpu().numpy(), transformed_points.detach().cpu().numpy(), "%s/pcs_%s.ply"%(exp_name, rate))
                save_pcs(source_points_2d.transpose(1,0).detach().cpu().numpy(), target_points_2d.transpose(1,0).detach().cpu().numpy(), transformed_points_2d.transpose(1,0).detach().cpu().numpy(), "%s/proj_pcs_%s.ply"%(exp_name, rate))

        all_cd.append(loss_3d.detach().cpu().numpy())

        if loss_3d.detach().cpu().numpy() < min_cd:
            min_cd = loss_3d.detach().cpu().numpy()
            sel_rate = rate

            random_metrix_best = random_metrix
            
            final_r, final_t, final_s = model.get_transform()
            np_r, np_t, np_s = final_r.detach().cpu().numpy(), final_t.detach().cpu().numpy(), final_s.detach().cpu().numpy()

    mesh = transform_mesh_o3d(mesh, random_metrix_best.T)

    mesh.scale(target_scale, np.array([0.0, 0.0, 0.0]))
    mesh.translate(target_center)

    mesh = transform_mesh_o3d(mesh, np_r.T)
    mesh.translate(np_t)
    mesh.scale(np_s, np.array([0.0, 0.0, 0.0]))

    return mesh, loss_3d.detach().cpu(), loss_2dmatching.detach().cpu()
