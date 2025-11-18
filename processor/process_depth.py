import os
import argparse
import sys

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))  
sys.path.append(root_path)
import PIL.Image
import pandas as pd
import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images


def vggt_infer_depth(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    # Load and preprocess example images (replace with your own image paths)
    image = load_and_preprocess_images([image_path]).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device, dtype=dtype):
            image = image[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(image)
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, image, ps_idx)
        # depth_map = depth_map.squeeze().cpu().numpy()
        return depth_map
    
def get_image_depth(image_path, use_gt=False):
    image_path = "O3DVQA" + image_path.split("O3DVQA")[1]
    image_path = image_path.replace('\\', '/')
    if use_gt:
        depth_file_path = image_path.replace('RGBVis', 'Depth').replace('rgb', 'depth').replace('.png', '.npy').replace('.jpg', '.npy')
        depth_map = np.load(depth_file_path) # shape (480, 640)
        depth_map = PIL.Image.fromarray(depth_map.squeeze())
    else:
        rgb_img = PIL.Image.open(image_path) # shape (480, 640)
        depth_map = vggt_infer_depth(image_path)
        depth_map = depth_map.squeeze().cpu().numpy() # shape (392, 518)
        depth_map = PIL.Image.fromarray(depth_map.squeeze())
    # print(depth_map.size)
    return depth_map


def main(output_dir, use_gt=False):
    if use_gt:
        print("Using ground truth in depth process module.")
    for filename in os.listdir(output_dir):
        if filename.endswith('.pkl'):
            pkl_path = os.path.join(output_dir, filename)
            # Load the DataFrame from the .pkl file
            df = pd.read_pickle(pkl_path)
            # Process each image and add the results to a new column
            df['depth_map'] = df['image_path'].apply(get_image_depth, args=[use_gt])

            # Save the updated DataFrame back to the .pkl file
            df.to_pickle(pkl_path)
            print(f"Processed and updated {filename}")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Process images from .pkl files", add_help=True)
    parser.add_argument("--output_dir", type=str, help="path to directory containing .pkl files", required=True)
    parser.add_argument("--use_gt", action='store_true', help="use the gt depth")
    args = parser.parse_args()

    main(args.output_dir, args.use_gt)

