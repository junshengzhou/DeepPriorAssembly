#!/bin/bash


output_path="data/outputs"
input_path="data/inputs"
image_id="479d2d66-4d1a-47ca-a023-4286fc547301---rgb_0017"

gpu_id="0"

#-------------------------------------------------Grounded-SAM for Segmentation-----------------------------------------------

eval "$(conda shell.bash hook)" 
conda activate grounded_sam
cd grounded_sam 

python segment_scenes.py --input_path "$input_path" --save_path "$output_path/segmentation" --image_id "$image_id" 

cd .. 


# -------------------------------------------StableDiffusion and CLIP for Inpainting---------------------------------------------

eval "$(conda shell.bash hook)" 
conda activate stablediffusion
cd stablediffusion

CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/img2img_inpainting.py --input_path "$output_path/segmentation" --outdir "$output_path/inpainting" --n_samples 6 --strength 0.5 --image_id "$image_id" --ckpt checkpoints/v2-1_512-ema-pruned.ckpt 

cd .. 

# -------------------------------------------------Shap-E for Object Generation---------------------------------------------

eval "$(conda shell.bash hook)"  
conda activate shap-e
cd shap-e 

CUDA_VISIBLE_DEVICES="$gpu_id" python object_generation.py --input_path "$output_path/inpainting" --output_path "$output_path/object_generation" --image_id "$image_id"

cd .. 

# -----------------------------------------------Dust3R for Geometry Estimation---------------------------------------------

eval "$(conda shell.bash hook)" 
conda activate dust3r
cd dust3r 

CUDA_VISIBLE_DEVICES="$gpu_id" python gen_scene_geometry.py --input_path "$input_path" --output_path "$output_path" --image_id "$image_id"

cd .. 

# -----------------------------------------------Final Registration by Optimization---------------------------------------------

eval "$(conda shell.bash hook)" 
conda activate dust3r
cd registration

CUDA_VISIBLE_DEVICES="$gpu_id" python optimization_5dof.py --image_id "$image_id" --geometry_dir "$output_path/geometry" --mask_dir "$output_path/segmentation" --object_dir  "$output_path/object_generation" --output_dir "$output_path/final_registration" 
