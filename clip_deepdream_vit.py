"""
CLIP-ViT DeepDream

Inspired by Alexander Mordvintsev's original DeepDream
https://www.tensorflow.org/tutorials/generative/deepdream

by zer0int
https://github.com/zer0int

See also (for more and different ViT dreams):
https://github.com/zer0int/CLIP-DeepDream
====================================================
"""

import os
import shutil
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import collections
import argparse
import random
import math
import glob
from colorama import Fore, Style

# CLIP
import clip
from clip.model import QuickGELU

# Custom imports
from cliptools import TotalVariation, CrossEntropyLoss, MatchBatchNorm, BaseFakeBN, LayerActivationNorm
from cliptools import ActivationNorm, NormalVariation, ColorVariation, fix_random_seed
from cliptools import NetworkPass
from cliptools import LossArray, TotalVariation
from cliptools import ViTFeatHook, ViTEnsFeatHook
from cliptools import TotalVariation as BaseTotalVariation, FakeColorDistribution as AbstractColorDistribution
from cliptools import FakeBatchNorm as BaseFakeBN, NormalVariation as BaseNormalVariation
from cliptools import ColorVariation as BaseColorVariation
from cliptools import ViTAttHookHolder, ViTGeLUHook, ClipGeLUHook, SpecialSaliencyClipGeLUHook
from cliptools import Clip, Tile, Jitter, RepeatBatch, ColorJitter, fix_random_seed
from cliptools import GaussianNoise
from cliptools import ClipViTWrapper as ClipWrapper
from cliptools import new_init, save_intermediate_step, save_image, fix_random_seed

# Stop spam regarding torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 


device="cuda" if torch.cuda.is_available() else "cpu"
base_folder = 'tmp_dd'
os.makedirs(base_folder, exist_ok=True)
out_folder = 'vit_deepdream'
os.makedirs(out_folder, exist_ok=True)
tmp_folder = 'tmp_dd/steps'
os.makedirs(tmp_folder, exist_ok=True)
steps_folder = 'tmp_dd/dream'
os.makedirs(steps_folder, exist_ok=True)
dreamtmp_folder = 'tmp_dd/tiled_in'
os.makedirs(dreamtmp_folder, exist_ok=True)
filename = tmp_folder


def parse_arguments():
    parser = argparse.ArgumentParser(description='DeepDream with CLIP ViT')
    parser.add_argument('--img', type=str, default=None, help="Path to a single input image")
    parser.add_argument('--model', default="ViT-L/14", help="Name or Path to a CLIP ViT model, default: ViT-L/14")
    parser.add_argument('--start_layer', type=int, default=19, help='Run deepdream from including layer [default: 19]')
    parser.add_argument('--end_layer', type=int, default=21, help='Run deepdream up to including layer [default: 21]')
    parser.add_argument('--num_feats', type=int, default=3, help='Top features per layer to visualize [default: 3]')
    parser.add_argument('--steps', type=int, default=300, help='Total optimization steps [default: 300]')
    parser.add_argument('--tv', type=float, default=1.0, help='Total Variation loss [default: 1.0]')
    parser.add_argument('--coeff', type=float, default=0.0005, help='Coefficient for TV [default: 0.0005]')
    parser.add_argument('--lr', type=float, default=0.07, help='Learning Rate [default: 0.07]')
    parser.add_argument("--no_cleanup", action='store_true', help="If set, retains temporary files. Default: Keep only final deepdream")
    parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (CUDA backends, torch, numpy)")
    return parser.parse_args()

args = parse_arguments()

if args.deterministic:
    fix_random_seed()

print(Fore.BLUE + Style.BRIGHT + "\n\n-----------------------------" + Fore.RESET)
print(Fore.BLUE + Style.BRIGHT + "    DeepDream for CLIP ViT" + Fore.RESET)
print(Fore.BLUE + Style.BRIGHT + "-----------------------------\n" + Fore.RESET)

class ImageNetVisualizer:
    def __init__(self, loss_array: LossArray, pre_aug: nn.Module = None,
                 post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, save_every: int = 200, saver: bool = True,
                 print_every: int = 5, **_):
        self.loss = loss_array
        self.saver = saver


        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None, layer: int = None, feature: int = None, clipname: str = None):
        if not img.is_cuda or img.device != torch.device('cuda:0'):
            img = img.to('cuda:0')
        if not img.requires_grad:
            img.requires_grad_()
        
        # ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']        
        optimizer = optimizer if optimizer is not None else optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{self.loss.header()}', flush=True)

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f'{i}\t{self.loss}', flush=True)
            if i % self.save_every == 0 and self.saver is True:
                save_intermediate_step(img, i, layer, feature, clipname, filename)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()

        optimizer.state = collections.defaultdict(dict)
        return img, filename

def get_clip_dimensions(clipmodel):
    model, preprocess = clip.load(clipmodel)
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            break
    num_layers = None
    num_features = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features
    return input_dims, num_layers, num_features

def load_clip_model(clipmodel, device='cuda'):
    model, _ = clip.load(clipmodel, device=device)
    model = ClipWrapper(model).to(device)
    return model

def generate_visualizations(model, clipname, layer_range_str, feature_range_str, image_size, tv, lr, steps, print_every, save_every, saver, coefficient):
    layer_range = parse_range(layer_range_str)
    feature_range = parse_range(feature_range_str)

    for layer in layer_range:
        for feature in feature_range:
            print(Fore.GREEN + Style.BRIGHT + f"Generating visualization for Layer {layer}, Feature {feature}..." + Fore.RESET)
            loss = LossArray()
            loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
            loss += TotalVariation(2, image_size, coefficient * tv)

            pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                            GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
            image = new_init(image_size, 1)

            visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient)
            image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

            save_image(image, f'{out_folder}/{clipname}_L{layer}_F{feature}.png')

def delete_files_in_subfolders(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(foldername, filename)
            try:
                os.remove(file_path)
            except FileNotFoundError:
                pass
            #print(f"Deleted: {file_path}")

def delete_subfolders(path):
    try:
        shutil.rmtree(path)
        print(f"\n[INFO] Successfully cleaned up {path}")
    except FileNotFoundError:
        pass
    except Exception as e:
        pass


def reassemble_tiles(tiles_folder, input_dim, output_filename):
    tile_files = [f for f in os.listdir(tiles_folder) if "tile_" in f and (f.endswith(".png") or f.endswith(".jpg"))]
    
    tile_positions = []
    for tile_file in tile_files:
        try:
            base_name = os.path.splitext(tile_file)[0]
            parts = base_name.split("_tile_")[1].split("_")
            row = int(parts[0])
            col = int(parts[1])
            tile_positions.append((row, col, tile_file))
        except (IndexError, ValueError) as e:
            print(Fore.RED + Style.BRIGHT + f"[WARNING] Skipping file due to parsing error: {tile_file}" + Fore.RESET)
            continue
    
    if not tile_positions:
        raise ValueError(Fore.RED + Style.BRIGHT + f"[WARNING] No valid tiles found in {tiles_folder}. Ensure filenames include '_tile_<row>_<col>'." + Fore.RESET)

    tile_positions.sort(key=lambda x: (x[0], x[1]))
    max_row = max(tile[0] for tile in tile_positions) + 1
    max_col = max(tile[1] for tile in tile_positions) + 1
    
    print(Fore.YELLOW + Style.BRIGHT + f"\n[INFO] Reassembling grid of size: {max_row}x{max_col}" + Fore.RESET)
    
    tiles = {}
    for row, col, tile_file in tile_positions:
        tile_path = os.path.join(tiles_folder, tile_file)
        tiles[(row, col)] = Image.open(tile_path)
    
    full_width = max_col * input_dim
    full_height = max_row * input_dim
    full_image = Image.new('RGB', (full_width, full_height))
    
    for (row, col), tile in tiles.items():
        left = col * input_dim
        upper = row * input_dim
        full_image.paste(tile, (left, upper))
    
    full_image.save(output_filename)
    print(Fore.GREEN + Style.BRIGHT + f"[INFO] Reassembled image saved to {output_filename}" + Fore.RESET)


def scale_and_tile_image(original_filename, input_dim, dreamtmp_folder):
    img = Image.open(original_filename)
    width, height = img.size
    
    min_upscale = 2 * input_dim
    aspect_ratio = width / height
    
    if aspect_ratio >= 1:  # Width >= Height
        upscaled_width = math.ceil(min_upscale * aspect_ratio / input_dim) * input_dim
        upscaled_height = min_upscale
    else:  # Height > Width
        upscaled_height = math.ceil(min_upscale / aspect_ratio / input_dim) * input_dim
        upscaled_width = min_upscale
    
    upscaled_width = math.ceil(upscaled_width / input_dim) * input_dim
    upscaled_height = math.ceil(upscaled_height / input_dim) * input_dim
    img_resized = img.resize((upscaled_width, upscaled_height))
    
    num_tiles_width = upscaled_width // input_dim
    num_tiles_height = upscaled_height // input_dim
    tile_width = input_dim
    tile_height = input_dim

    print(f"\n[INFO] Upscaled dimensions: {upscaled_width}x{upscaled_height}")
    print(f"[INFO] Number of tiles: {num_tiles_width}x{num_tiles_height}")
    
    # Cut the image into tiles and save each
    for row in range(num_tiles_height):  # Iterate over height (rows)
        for col in range(num_tiles_width):  # Iterate over width (columns)
            # Define the bounding box for the current tile
            left = col * tile_width
            upper = row * tile_height
            right = left + tile_width
            lower = upper + tile_height
            bbox = (left, upper, right, lower)
            
            # Crop the image to the bounding box to create the tile
            img_tile = img_resized.crop(bbox)
            
            # Construct the filename for the tile (deterministic based on row, col, and original filename)
            base_name = os.path.splitext(os.path.basename(original_filename))[0]
            tile_filename = f"{base_name}_tile_{row}_{col}.png"
            
            # Save the tile to the specified directory
            img_tile.save(os.path.join(dreamtmp_folder, tile_filename))

 
def generate_deepdream(model, clipname, layer, feature, image_size, tv, lr, steps, print_every, save_every, saver, coefficient):
    loss = LossArray()
    loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient * tv)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip() 
  
    preprocess = Compose([
        Resize((input_dims, input_dims)), #224, 336
        ToTensor(), 
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

    folder_path = dreamtmp_folder
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img_name = os.path.splitext(os.path.basename(image_path))[0]
            filename = img_name
            image = Image.open(image_path)
            image = image.convert("RGB")
            image = preprocess(image)
            image = image.unsqueeze(0).to(device)    

            # Extract row and col from img_name (e.g., "filename_tile_0_1.png")
            if "_tile_" in img_name:
                try:
                    parts = img_name.split("_tile_")[1].split("_")
                    row = int(parts[0])
                    col = int(parts[1])
                except (IndexError, ValueError):
                    print(Fore.RED + Style.BRIGHT + f"[WARNING] Error parsing row/col for {img_name}" + Fore.RESET)
                    continue
            else:
                print(Fore.RED + Style.BRIGHT + f"[WARNING] Skipping file without '_tile_' in name: {img_name}" + Fore.RESET)
                continue           
            
            visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient)
            modified_image, filename = visualizer(image, layer=layer, feature=feature, clipname=clipname)
            image.data = modified_image.data
            dreamy_folder = f"{steps_folder}/L{layer}_F{feature}/"
            os.makedirs(dreamy_folder, exist_ok=True)
            filename = f"{layer}_{feature}_{img_name}_tile_{row}_{col}.png"
            save_image(image, f"{steps_folder}/L{layer}_F{feature}/{filename}")

class ClipNeuronCaptureHook:
    def __init__(self, module: torch.nn.Module, layer_idx: int):
        self.layer_idx = layer_idx
        self.activations = None
        self.top_values = None
        self.top_indices = None
        self.hook_handle = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.activations = output.detach()

    def get_top_neurons(self, k=5):
        if self.activations is not None:
            self.top_values, self.top_indices = torch.topk(self.activations, k, dim=-1)
            return self.layer_idx, self.top_values, self.top_indices
        return None, None, None
    
    def remove(self):
        self.hook_handle.remove()

def register_cliptools(model, num_layers):
    cliptools = []
    layer_idx = 0
    for name, module in model.named_modules():
        if isinstance(module, QuickGELU):
            hook = ClipNeuronCaptureHook(module, layer_idx)
            cliptools.append(hook)
            layer_idx += 1
            if layer_idx >= num_layers:
                break
    return cliptools

def get_all_top_neurons(cliptools, k=5):
    all_top_neurons = []
    for hook in cliptools:
        layer_idx, top_values, top_indices = hook.get_top_neurons(k)
        if top_values is not None:
            all_top_neurons.append((layer_idx, top_values, top_indices))
    return all_top_neurons  

def get_clipname(clipmodel):
    # Known model names, could be expanded as needed
    known_models = {'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'}

    # Heuristic: If clipmodel contains a path separator or endswith known model extension, treat as path
    is_path = (
        '/' in clipmodel or '\\' in clipmodel
        or clipmodel.endswith('.pt') or clipmodel.endswith('.pth') or clipmodel.endswith('.bin') or clipmodel.endswith('.safetensors')
    ) and clipmodel not in known_models

    if is_path:
        basename = os.path.basename(clipmodel)
        clipname, _ = os.path.splitext(basename)
        return clipname
    else:
        return clipmodel.replace("/", "-").replace("@", "-")

    
# ['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
clipmodel = args.model
clipname = get_clipname(clipmodel)

input_dims, num_layers, num_features = get_clip_dimensions(clipmodel)
print(f"\n[INFO] Selected input dimension for {clipmodel}: {input_dims}")
print(Fore.GREEN + Style.BRIGHT + f"[INFO] Number of Layers: {num_layers} with {num_features} Features / Layer\n" + Fore.RESET)

sideX = input_dims
sideY = input_dims

transforming = transforms.Compose([
  transforms.Resize((sideX, sideY)),
  transforms.ToTensor(),
  transforms.Lambda(lambda x: x[:3, :, :]),  # Ensure RGB only
  transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
])


image_path = args.img
original_filename = image_path
dreamimage = image_path
proc_folder = dreamtmp_folder
os.makedirs(proc_folder, exist_ok=True)
save_img_name = os.path.splitext(os.path.basename(image_path))[0]


  
def main():
    model = load_clip_model(clipmodel)
    image_size = input_dims
    
    img = Image.open(image_path)
    input_image = transforming(img).unsqueeze(0).to(device)

    cliptools = register_cliptools(model, num_layers)
    _ = model(input_image)
    
    # Retrieve top neurons across all layers
    all_top_neurons = get_all_top_neurons(cliptools, k=5)
    top_features_per_layer = {}
    for layer_idx, _, top_indices in all_top_neurons:
        feature_indices = top_indices[0][0].cpu().tolist()
        top_features_per_layer[layer_idx] = feature_indices  
    for hook in cliptools:
        hook.remove()
        
    start_layer = args.start_layer
    end_layer = args.end_layer
    
    if args.num_feats > 5 or args.num_feats < 1:
        print(Fore.YELLOW + Style.BRIGHT + f"[WARNING] Argument '--num_feats' cannot be >5 / <1, but is set to {args.num_feats}!\nSetting a balanced num_feats=3 and continuing..." + Fore.RESET)
        num_features_to_visualize = 3
    else:
        num_features_to_visualize = args.num_feats
    
    # ---- Visualization settings
    tv = args.tv
    coefficient = args.coeff
    lr = args.lr
    steps = args.steps
    print_every = 10
    save_every = 50
    saver = True
    # ----

    # Generate visualizations for top features of selected layers
    original_filename = image_path
    input_dim = input_dims
    filename = "filename"
    filesclean = glob.glob(f'{dreamtmp_folder}/*')
    for f in filesclean:
        os.remove(f)
    scale_and_tile_image(original_filename, input_dim, dreamtmp_folder)
    for layer_idx, feature_indices in top_features_per_layer.items():
        if start_layer <= layer_idx <= end_layer:
            for feature_idx in feature_indices[:num_features_to_visualize]:
                print(Fore.GREEN + Style.BRIGHT + f"\nGenerating for Layer {layer_idx}, Feature {feature_idx}:" + Fore.RESET)
                generate_deepdream(model, clipname, layer_idx, feature_idx, image_size, tv, lr, steps, print_every, save_every, saver, coefficient)
                reassemble_tiles(f"{steps_folder}/L{layer_idx}_F{feature_idx}/", 224, f"{out_folder}/{save_img_name}_L{layer_idx}_F{feature_idx}_full.png")
    if args.no_cleanup:
        pass
    else:
        delete_files_in_subfolders(base_folder)
        delete_subfolders(base_folder)
        print(Fore.YELLOW + Style.BRIGHT + f"[INFO] Cleaned up intermediate files and deleted '{base_folder}'. Set '--no_cleanup' to keep temporary files." + Fore.RESET)
        print(Fore.GREEN + Style.BRIGHT + f"\n[DONE] - Check the folder '{out_folder}' for the results!" + Fore.RESET)

if __name__ == '__main__':
    main()