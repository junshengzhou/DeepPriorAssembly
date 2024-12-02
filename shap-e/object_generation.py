import torch
import os
from shap_e.diffusion.sample import sample_latents
from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
from shap_e.models.download import load_model, load_config
from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
from shap_e.util.image_util import load_image
import os
import argparse, os
from PIL import Image

import sys

def list_files_in_folder(folder_path):
    img_path_all = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if 'filter_00000' not in file_path and 'filter_00001' not in file_path and 'filter_00002' not in file_path:
                continue
            img_path_all.append(file_path)
    return img_path_all

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
xm = load_model('transmitter', device=device)
model = load_model('image300M', device=device)
diffusion = diffusion_from_config(load_config('diffusion'))
batch_size = 2
guidance_scale = 3.0



opt.input_path = opt.input_path + '/%s' % (opt.image_id)
img_path_all = list_files_in_folder(opt.input_path)


# To get the best result, you should remove the background and show only the object of interest to the model.
# image = load_image("example_data/chair.png")

out_dir_base = opt.output_path

for i in range(len(img_path_all)):
    image = load_image(img_path_all[i])
    image = image.resize((320, 240),Image.ANTIALIAS)
    ind = img_path_all[i].split('/')[-1].split('.')[0]

    out_dir = '%s/%s/%s/%s' % (out_dir_base, img_path_all[i].split('/')[-4], img_path_all[i].split('/')[-3], img_path_all[i].split('/')[-1].split('.')[0])

    os.makedirs(out_dir, exist_ok=True)

    latents = sample_latents(
        batch_size=batch_size,
        model=model,
        diffusion=diffusion,
        guidance_scale=guidance_scale,
        model_kwargs=dict(images=[image] * batch_size),
        progress=True,
        clip_denoised=True,
        use_fp16=True,
        use_karras=True,
        karras_steps=64,
        sigma_min=1e-3,
        sigma_max=160,
        s_churn=0,
    )
    render_mode = 'nerf' # you can change this to 'stf' for mesh rendering
    size = 64 # this is the size of the renders; higher values take longer to render.


    from shap_e.util.notebooks import decode_latent_mesh


    for i, latent in enumerate(latents):
        t = decode_latent_mesh(xm, latent).tri_mesh()

        with open(f'{out_dir}/mesh_{i}.obj', 'w') as f:
            t.write_obj(f)