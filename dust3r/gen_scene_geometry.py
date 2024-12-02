from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.image import load_images
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
import torch
import numpy as np
import os
from skimage.transform import resize
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_path",
    type=str,
    default="test",
    help="path to the input image"
)

parser.add_argument(
    "--output_path",
    type=str,
    default="test",
    help="the prompt to render"
)

parser.add_argument(
    "--image_id",
    type=str,
    default="test",
    help="the prompt to render"
)


opt = parser.parse_args()

rotation_matrix = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]
])

def norm_pc(point_cloud):
    min_vals = np.min(point_cloud, axis=0)
    max_vals = np.max(point_cloud, axis=0)

    bbox_size = max_vals - min_vals 
    max_edge = np.max(bbox_size)    

    scale = 2.0 / max_edge        
    center = (max_vals + min_vals) / 2  

    return center, scale

if __name__ == '__main__':
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 300

    model_name = "checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth"
    # you can put the path to a local checkpoint in model_name if needed
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name).to(device)
    # load_images can take a list of images or a directory

    input_image = os.path.join(opt.input_path, opt.image_id + '.png')
    images = load_images([input_image, input_image], size=512)
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=batch_size)

    # at this stage, you have the raw dust3r predictions
    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()



    scene_pc_np = pts3d[0].reshape(-1, 3).detach().cpu().numpy()
    pts3d_np = pts3d[0].detach().cpu().numpy()


    output_path_pcs = os.path.join(opt.output_path, 'geometry', opt.image_id.split('.')[0])
    os.makedirs(output_path_pcs, exist_ok=True)

    np.save(os.path.join(output_path_pcs, 'focal.npy'), focals.detach().cpu().numpy())


    seg_info = np.load(os.path.join(opt.output_path, 'segmentation', opt.image_id.split('.')[0], 'mask_all.npy'))
    expanded_seg_info = np.array([resize(instance, (384, 512), order=0, preserve_range=True, anti_aliasing=False) for instance in seg_info], dtype=bool)


    results = [pts3d_np[expanded_seg_info[i]] for i in range(expanded_seg_info.shape[0])]

    center, scale = norm_pc(np.vstack(results))

    np.savetxt(os.path.join(output_path_pcs,'whole_pc.xyz'), scene_pc_np)

    for idx, result in enumerate(results):
        np.savetxt(os.path.join(output_path_pcs,'instance_%d.xyz' % idx), result)
        print(f"Instance {idx + 1}: Extracted {result.shape} elements")

