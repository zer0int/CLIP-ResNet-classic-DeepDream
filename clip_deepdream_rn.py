
"""
CLIP-ResNet DeepDream

Inspired by Alexander Mordvintsev's original DeepDream
https://www.tensorflow.org/tutorials/generative/deepdream

by zer0int
https://github.com/zer0int
====================================================
"""

import os
import argparse
import random
import numpy as np
from typing import Tuple, List
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
from colorama import Fore, Style

import matplotlib.pyplot as plt
from scipy.ndimage import filters

import kornia.augmentation as kaugs
import kornia
import torchvision
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
import copy

import clip
from cliptools import fix_random_seed

# Stop spam regarding torch
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# ------------- CLI -----------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--img', required=True, help='The path/to/image.png (or .jpg) to DeepDream on')
parser.add_argument('--model', default='RN50x4', choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64'], help='CLIP Model to use for DeepDream')
parser.add_argument('--mode', default='m_ch_norm', choices=['s_neuron', 's_channel', 'm_ch_mean', 'm_ch_sim', 'm_ch_norm', 'm_ch_sim_norm'], help='Visualization mode; default: m_ch_norm')
parser.add_argument('--layers', nargs='+', help=("Space-separated list of target layers [default: layer3], e.g.: --layers layer2 layer3"))
parser.add_argument('--iters', type=int, default=400, help='Optimization steps per octave')
parser.add_argument('--octaves', type=int, nargs='+', default=[-3,-2,-1,0,1,2,3], help='Octave indices; default: -3 to +3; for smaller image and less compute, use --octaves -2 -1 0 1 2')
parser.add_argument('--octave_scale', type=float, default=1.5, help='Octave scaling factor; default: 1.5; for octaves -2 to +2, try: --octave_scale 1.3')
parser.add_argument('--lr', type=float, default=0.01, help='Base learning-rate per step; default: 0.01; for s_neuron, try 0.03')
parser.add_argument('--k', type=int, default=10, help='Top-k features (channels or neurons) per layer to visualize')
parser.add_argument('--batch_size_ga', type=int, default=10, help="Batch size for Gradient Ascent [GA] on Text, '--mode *_neuron' only. Also set --k to the same!")
parser.add_argument("--no_reload_ga", action='store_true', help="If set, always re-computes [GA] embeddings; default: Load from file, if exists")
parser.add_argument("--filter_acts", action='store_true', help="Filter activations against always active for any image; --mode s_* only")
parser.add_argument("--filter_peaky", action='store_true', help="Filter for 'peakiness'; --mode s_* only")
parser.add_argument("--acts_rand", action='store_true', help="Choose random highly salient activation; --mode s_* only")
parser.add_argument("--lr_by_acts", action='store_true', help="Scale LR relative to idx activation value; --mode s_* only")
parser.add_argument("--deterministic", action='store_true', help="Use deterministic behavior (CUDA backends; torch, numpy)")
parser.add_argument("--save_steps", action='store_true', help="Save intermediate visualization steps (every 20th)")
args = parser.parse_args()

if args.deterministic:
    fix_random_seed()

if args.layers:
    target_layers: List[str] = args.layers
else:
    args.layers = ['layer3']
    target_layers = ['layer3']

scaler = GradScaler()
print(Fore.BLUE + Style.BRIGHT + "\n\n-----------------------------" + Fore.RESET)
print(Fore.BLUE + Style.BRIGHT + "  DeepDream for CLIP ResNet" + Fore.RESET)
print(Fore.BLUE + Style.BRIGHT + "-----------------------------\n" + Fore.RESET)

# ------------- Pre-/postprocessing -------------------------------------------
def get_clip_dim(preprocess) -> int:
    """
    Extracts the CLIP input dimension from the preprocess pipeline.
    Returns the dimension as int (e.g., 224, 336, 384).
    """
    for t in preprocess.transforms:
        if hasattr(t, 'size'):
            s = t.size
            if isinstance(s, int):
                return s
            elif isinstance(s, (tuple, list)):
                if s[0] == s[1]:
                    return s[0]
                else:
                    # If non-square (??), return the max/min as needed
                    return max(s)
    raise ValueError("Could not find input dimension from preprocess transforms.")

def to_pil(t: torch.Tensor) -> Image.Image:
    inv_normalize = transforms.Normalize(
        mean=[-m / s for m, s in zip((0.48145466, 0.4578275, 0.40821073),
                                     (0.26862954, 0.26130258, 0.27577711))],
        std=[1 / s for s in (0.26862954, 0.26130258, 0.27577711)]
    )
    t = inv_normalize(t.squeeze(0).cpu()).clamp(0, 1)
    return transforms.ToPILImage()(t)

# ------------- Helpers -------------------------------------------------------
def get_visual_submodel(clip_model, target_layer: str) -> torch.nn.Sequential:
    """
    Returns a torch.nn.Sequential running from pixel-space up to (and including) `target_layer`.
    Dynamically detects the layer order and names from the model.
    """
    visual = clip_model.visual
    layers = [
        visual.conv1, visual.bn1, visual.relu1,
        visual.conv2, visual.bn2, visual.relu2,
        visual.conv3, visual.bn3, visual.relu3,
        visual.avgpool
    ]
    
    # Find all "layer{i}" modules present in the visual model
    layer_names = []
    for name, module in visual.named_children():
        if name.startswith("layer") and name[5:].isdigit():
            layer_names.append(name)
    layer_names.sort(key=lambda x: int(x[5:]))  # Sort numerically by index

    for lname in layer_names:
        layers.append(getattr(visual, lname))
        if lname == target_layer:
            break
    else:
        raise ValueError(f"Target layer '{target_layer}' not found among {layer_names}.")

    return torch.nn.Sequential(*layers)

def ensure_clip_square(img: torch.Tensor, clip_dim: int) -> torch.Tensor:
    _, _, h, w = img.shape
    if h != w or h != clip_dim:
        # Center crop to min(h, w), then resize to (clip_dim, clip_dim)
        min_dim = min(h, w)
        img = torchvision.transforms.functional.center_crop(img, min_dim)
        img = torch.nn.functional.interpolate(img, size=(clip_dim, clip_dim), mode='bilinear', align_corners=False)
    return img

def get_model_text_embed_dim(model):
    # Get the expected text embedding dim for this model (CLIP standard)
    with torch.no_grad():
        dummy = clip.tokenize(["dummy"]).to(next(model.parameters()).device)
        out = model.encode_text(dummy)
    return out.shape[1]


# --------------------------------------------------------------------------------
# ------------- GradCAM ----------------------------------------------------------
# --------------------------------------------------------------------------------

def get_attnpool_attention_map(model, img):
    # Register a hook to access the attention weights @ AttentionPool2d
    attn_maps = []
    def hook_fn(module, input, output):
        attn = module.attn.last_attn_weights
        attn_maps.append(attn.detach().cpu())
    h = model.visual.attnpool.register_forward_hook(hook_fn)
    with torch.no_grad():
        model.visual(img)
    h.remove()
    return attn_maps[0]

class AttnHook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)
        
    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()
        
    @property
    def activation(self) -> torch.Tensor:
        return self.data
    
    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad
"""
Taken and adapted from https://github.com/kevinzakka/clip_playground
"""
def gradCAM(
    model: nn.Module,
    input: torch.Tensor,
    target: torch.Tensor,
    layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()
        
    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)
        
    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with AttnHook(layer) as hook:        
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()
    
        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

   
    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])
        
    return gradcam

def do_gradcam(model, inp_img, text_embeds, layer='layer4'):
    """
    For each embedding in text_embeds, run GradCAM, return list of (y, x) coordinates.
    """
    device = inp_img.device
    gradcam_locs = []
    for i in range(text_embeds.shape[0]):
        text_embed = text_embeds[i].unsqueeze(0).to(dtype=model.dtype, device=device)
        attn_map = gradCAM(
            model.visual,
            inp_img,
            text_embed,
            getattr(model.visual, layer)
        )
        attn_map = attn_map.squeeze().detach().cpu()
        max_idx = torch.argmax(attn_map)
        #max_y, max_x = np.unravel_index(max_idx.cpu().numpy(), attn_map.shape)
        #gradcam_locs.append((max_y, max_x))
        max_y, max_x = np.unravel_index(max_idx.cpu().numpy(), attn_map.shape)

        # -------- store relative coordinates --------
        rel_y = max_y / (attn_map.shape[0] - 1)
        rel_x = max_x / (attn_map.shape[1] - 1)
        gradcam_locs.append((rel_y, rel_x))

        print(f"Most salient neuron for {i} at (y, x): {max_y}, {max_x}")
    return gradcam_locs


# --------------------------------------------------------------------------------
# ------------- Gradient Ascent Text Embeddings ----------------------------------
# --------------------------------------------------------------------------------
"""
Uses a heavily modified version of Original CLIP Gradient Ascent Script: by Twitter / X: @advadnoun
"""
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # Expect mean and std as lists of 3 elements.
        self.register_buffer('mean', torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x):
        return (x - self.mean) / self.std

# Image Loader
def load_image(img_path, sideX, sideY):
    im = torch.tensor(np.array(Image.open(img_path).convert("RGB"))).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255
    im = F.interpolate(im, (sideX, sideY))
    return im

# Augmentation Pipeline
def augment(into, augs):
    return augs(into)

# Gradient Ascent / text encoder forward
def clip_encode_text(model, text, many_tokens, prompt):
    x = torch.matmul(text, model.token_embedding.weight)
    x = x + model.positional_embedding
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x)
    x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ model.text_projection
    return x

# Entertain user by printing CLIP's 'opinion' rants about image to console
def checkin(loss, tx, lll, tok, bests, imagename):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('<|startoftext|>', '').replace('<|endoftext|>', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}" + Fore.RESET)
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = j.replace('\ufffd', '')
        tokens = j.split()
        unique_tokens.update(tokens)
    os.makedirs("txtopinion", exist_ok=True)
    with open(f"txtopinion/tokens_{imagename}.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))

# Softmax
class Pars(torch.nn.Module):
    def __init__(self, batch_size, many_tokens, prompt):
        super(Pars, self).__init__()
        self.batch_size = batch_size
        self.many_tokens = many_tokens
        self.prompt = prompt
        self.gumbel_temp = 1000

        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        self.prompt_embeddings = torch.zeros(batch_size, len(prompt), 49408).cuda()
        for jk, pt in enumerate(prompt):
            self.prompt_embeddings[:, jk, pt] = 1 

        pad_length = 77 - (self.many_tokens + len(self.prompt) + 1)
        self.pad = torch.zeros(self.batch_size, pad_length, 49408).cuda()
        self.pad[:, :, 49407] = 1

    def forward(self):
        soft = F.gumbel_softmax(self.normu, tau=self.gumbel_temp, dim=-1, hard=True)

        return torch.cat([self.start, self.prompt_embeddings, soft, self.pad], 1)


# Gradient Ascent
def ascend_txt(image, model, lats, many_tokens, prompt, nom, augment):
    iii = nom(augment(image[:,:3,:,:].expand(lats.normu.shape[0], -1, -1, -1)))
    iii = model.encode_image(iii).detach()
    lll = lats()
    tx = clip_encode_text(model, lll, many_tokens, prompt)
    loss = -100 * torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, lats.normu.shape[0]).T.mean(1)
    return loss, tx, lll


# Loop with AMP
def train(image, model, lats, many_tokens, prompt, optimizer, nom, augment):
    with autocast():
        loss1, tx, lll = ascend_txt(image, model, lats, many_tokens, prompt, nom, augment)
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward(retain_graph=True)
    scaler.step(optimizer)
    scaler.update()
    return loss1, tx, lll

def generate_target_text_embeddings(img_path, model, lats, optimizer, training_iterations, checkin_step, many_tokens, prompt, nom, augment, tok, bests):

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    img = load_image(img_path, model.visual.input_resolution, model.visual.input_resolution)
    print(Fore.YELLOW + Style.BRIGHT + f"\nRunning gradient ascent for {img_name}...\n" + Fore.RESET)

    scaler = GradScaler()

    best_loss = float('inf')  # Initialize the best loss as infinity
    best_text_embeddings = None  # Placeholder for the best text embeddings

    for j in range(training_iterations):
           
        loss, tx, lll = train(img, model, lats, many_tokens, prompt, optimizer, nom, augment)
        current_loss = loss.mean().item()

        # Update best embeddings if current loss is better
        if current_loss < best_loss:
            best_loss = current_loss
            best_text_embeddings = copy.deepcopy(tx.detach())
            print(Fore.RED + Style.BRIGHT + f"New best loss: {best_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)
            print(Fore.RED + Style.BRIGHT + "-------------------" + Fore.RESET)

        # Print learning rate for monitoring
        if j % 50 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(Fore.GREEN + f"Iteration {j}: Average Loss: {current_loss:.3f}" + Fore.RESET)
            checkin(loss, tx, lll, tok, bests, img_name)

    # Save the best embeddings
    os.makedirs("txtembeds", exist_ok=True)
    torch.save(best_text_embeddings, f"txtembeds/{img_name}_emb.pt")
    print(Fore.MAGENTA + Style.BRIGHT + "\nBest text embedding saved to 'txtembeds'.\nTokens (CLIP 'opinion') saved to 'txtopinion' folder.\n" + Fore.RESET)
    del optimizer, lats, scaler, prompt
    return img, best_text_embeddings, img_path

def run_gradient_ascent(model, preprocess, use_image):

    normalizer = Normalization([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]).cuda()
    model = model.float()
    
    tok = clip.simple_tokenizer.SimpleTokenizer()

    augs = torch.nn.Sequential(
        kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
    ).cuda()
    
    bests = {1000: 'None', 1001: 'None', 1002: 'None', 1003: 'None', 1004: 'None', 1005: 'None'}
    prompt = clip.tokenize('''''').numpy().tolist()[0]
    prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

    batch_size=args.batch_size_ga
    checkin_step = 10  
    iterations=300    
    tokinit = 4
    lats = Pars(batch_size, tokinit, prompt).cuda()    
    optimizer = torch.optim.Adam([{'params': [lats.normu], 'lr': 5}])

    img, target_text_embedding, img_path = generate_target_text_embeddings(use_image, model, lats, optimizer, iterations, checkin_step, tokinit, prompt, normalizer, augs, tok, bests)
    print(f"Done processing image: {img_path}")


# --------------------------------------------------------------------------------
# ------------- Activations dict and other helpers -------------------------------
# --------------------------------------------------------------------------------
activations = {}

def make_hook(key: str):
    """
    Forward-hook.
    Keeps the tensor **with its graph** so we can back-propagate.
    -> no .detach()
    """
    def _hook(_m, _inp, out):
        activations[key] = out
    return _hook

def random_roll(img: torch.Tensor, maxroll: int) -> Tuple[Tuple[int, int], torch.Tensor]:
    shift_x = random.randint(-maxroll, maxroll)
    shift_y = random.randint(-maxroll, maxroll)
    return (shift_x, shift_y), torch.roll(img, (shift_y, shift_x), (2, 3))

def _deepest_layer(layers: List[str]) -> str:
    """Return the numerically deepest layer name from ['layer2', 'layer4', …]."""
    return max(layers, key=lambda l: int(l[5:]))

def get_deepdream_channels_contrastive(model, img: torch.Tensor,
                                       layer_name: str,
                                       k: int = 10) -> List[Tuple[int, float]]:
    """
    Saliency = activation on real image – max(activation on gray, activation on noise)
    Returns top-k (channel_idx, saliency_val).
    """
    gray_val = img.mean()
    gray_img = torch.ones_like(img) * gray_val
    noise_img = torch.randn_like(img) * img.std() + img.mean()
    images = [img, gray_img, noise_img]

    # --- Hook ---
    layer_mod = getattr(model.visual, layer_name)
    submodel = get_visual_submodel(model, layer_name)

    acts_list = []
    print(Fore.GREEN + Style.BRIGHT + f"Getting salience for layer {layer_name}, k={k}" + Fore.RESET)
    
    if args.filter_acts:
        print(Fore.CYAN + "Using filter_acts mode (saliency = real - max(gray, noise))" + Fore.RESET)
        for im, desc in zip(images, ['real', 'gray', 'noise']):
            activations.clear()
            h = layer_mod.register_forward_hook(make_hook('feat'))
            with torch.no_grad():
                submodel(im)
            h.remove()
            acts = activations['feat'][0].mean((1, 2))  # [C]
            acts_list.append(acts)
            #print(f"  {desc} activations (mean over H,W): {acts.cpu().numpy()}")
        real, gray, noise = acts_list
        sal = real - torch.stack([gray, noise]).max(0).values
        #print(f"  Saliency (real - max(gray, noise)): {sal.cpu().numpy()}")
        print(Fore.BLUE + Style.BRIGHT + "-------------------" + Fore.RESET)

        if args.filter_peaky:
            # Spatial "peakiness" filter
            print("  Applying peaky filter")
            activations.clear()
            h = layer_mod.register_forward_hook(make_hook('feat'))
            with torch.no_grad():
                submodel(img)
            h.remove()
            acts_full = activations['feat'][0]
            peaky = acts_full.flatten(1).max(1)[0] / (acts_full.mean((1, 2)) + 1e-6)
            #print(f"  Peakiness ratio per channel: {peaky.cpu().numpy()}")
            before_count = (sal > float('-inf')).sum().item()
            sal[peaky <= 1.2] = float('-inf')
            after_count = (sal > float('-inf')).sum().item()
            print(f"    Filtered out {before_count - after_count} channels with low peakiness")
            print(Fore.BLUE + Style.BRIGHT + "-------------------" + Fore.RESET)
    elif args.acts_rand:
        print(Fore.CYAN + "Using acts_rand mode: pick k random channels among up-to-top-100 activations > 0.1" + Fore.RESET)
        activations.clear()
        h = layer_mod.register_forward_hook(make_hook('feat'))
        with torch.no_grad():
            submodel(img)
        h.remove()
        acts = activations['feat'][0].mean((1, 2))  # [C]
        #print(f"  Real activations (mean over H,W): {acts.cpu().numpy()}")
        # Filter channels with act > 0.1
        mask = acts > 2
        filtered_idxs = torch.where(mask)[0]
        filtered_vals = acts[filtered_idxs]
        print(f"  Channels with activation > 0.1: idxs={filtered_idxs.cpu().numpy()}, vals={filtered_vals.cpu().numpy()}")
        print(Fore.BLUE + Style.BRIGHT + "-------------------" + Fore.RESET)
        # Sort and pick up to top 100
        if len(filtered_idxs) > 0:
            top_acts, top_idxs = torch.topk(filtered_vals, min(100, filtered_vals.shape[0]))
            selected_idxs = filtered_idxs[top_idxs]
            print(f"  Top {min(100, filtered_vals.shape[0])} indices: {selected_idxs.cpu().numpy()}, vals={top_acts.cpu().numpy()}")
            # Pick k random indices among these
            rand_sel = random.sample(range(len(selected_idxs)), min(k, len(selected_idxs)))
            chosen_idxs = selected_idxs[rand_sel]
            chosen_vals = top_acts[rand_sel]
            print(f"  Randomly selected indices: {chosen_idxs.cpu().numpy()}, vals={chosen_vals.cpu().numpy()}")
            print(Fore.BLUE + Style.BRIGHT + "-------------------" + Fore.RESET)
            return [(i.item(), v.item()) for i, v in zip(chosen_idxs, chosen_vals)]
        else:
            print("  No activations above threshold, returning top-k regardless")
            vals, idxs = torch.topk(acts, k)
            print(f"  Chosen indices: {idxs.cpu().numpy()}, vals={vals.cpu().numpy()}")
            print(Fore.BLUE + Style.BRIGHT + "-------------------" + Fore.RESET)
            return [(i.item(), v.item()) for i, v in zip(idxs, vals)]
    else:
        print(Fore.CYAN + "Using plain activations mode: take top-k channels by mean activation (no contrastive images)" + Fore.RESET)
        print(Fore.RED + ">>> Tip! " + Fore.YELLOW + "Set --filter_acts to filter out generalizing (even for noise!) salient activations." + Fore.RESET)
        print(Fore.YELLOW + "The 'generalizers' usually show up for every image (no matter what) in the first top-k positions:" + Fore.RESET)
        activations.clear()
        h = layer_mod.register_forward_hook(make_hook('feat'))
        with torch.no_grad():
            submodel(img)
        h.remove()
        acts = activations['feat'][0].mean((1, 2))  # [C]
        #print(f"  Real activations (mean over H,W): {acts.cpu().numpy()}")
        print(Fore.BLUE + Style.BRIGHT + "-------------------" + Fore.RESET)
        sal = acts

    vals, idxs = torch.topk(sal, k)
    vals = torch.round(vals).to(torch.int)
    print(f"Selected top-k indices: {idxs.cpu().numpy()}\nActivation Values: {vals.cpu().numpy()}")
    print(Fore.BLUE + Style.BRIGHT + "-------------------" + Fore.RESET)
    return [(i.item(), round(v.item(), 2)) for i, v in zip(idxs, vals)]


def scale_salience(salient: List[Tuple[int, float]]) -> np.ndarray:
    """Map saliency values to discrete LR scaling factors ∈ [1, 10]."""
    vals = np.array([v for _, v in salient])
    mn, mx = vals.min(), vals.max()
    if np.isclose(mx, mn):
        return np.ones_like(vals, int)
    scale = 1 + (mx - vals) / (mx - mn) * 9
    return np.clip(scale.round(), 1, 10).astype(int)



# --------------------------------------------------------------------------------
# ------------- Gradient Ascent ~ DEEPDREAM MAIN ---------------------------------
# --------------------------------------------------------------------------------
def main():
    # ------------- Tiled gradient for multi-layer -------------------------------
    def tiled_gradient_multilayer(
            img: torch.Tensor,
            layers: List[str],
            tile_size: int,
            mode: str,
            idxs: List[int] = None,                    # per-layer feature indices
            xy_list: List[Tuple[int, int]] = None      # per-layer spatial coords
    ) -> torch.Tensor:
        """
        Accumulates gradient for an *arbitrary* set of layers.

        `idxs` and `xy_list` must be either None (mean-style objectives) or
        a list with the same length as `layers`.
        """
        yx_glo = None # Just a thing for filenames
        ch_idx_glo = None # Make sure no overwrites when saving
       
        b, c, h, w = img.shape
        grad = torch.zeros_like(img)
        (shift_x, shift_y), img_rolled = random_roll(img, tile_size // 2)

        deepest = _deepest_layer(layers)
        submodel = get_visual_submodel(model, deepest)

        activ_keys = {ln: ln for ln in layers}

        REQUIRED_MIN = {'layer2': 36, 'layer3': 18, 'layer4': 9}.get(deepest, 18)

        for y0 in range(0, h, tile_size):
            for x0 in range(0, w, tile_size):
                y1, x1 = min(y0 + tile_size, h), min(x0 + tile_size, w)
                if (y1 - y0) < REQUIRED_MIN or (x1 - x0) < REQUIRED_MIN:
                    continue

                tile = (img_rolled[..., y0:y1, x0:x1].detach().clone().requires_grad_(True))


                # --- Register hooks, run fwd, compute loss ---
                activations.clear()
                hooks = [getattr(model.visual, ln)
                             .register_forward_hook(make_hook(ln))
                         for ln in layers]

                submodel(tile)      # forward pass finished
                for hook in hooks:  # drop references
                    hook.remove()

                eps = 1e-7
                obj = 0.0

                # --- Available modes ---
                for i, ln in enumerate(layers):
                    act = activations[ln]
                    yx_glo = xy_list[i] if xy_list else None
                    ch_idx_glo = idxs[i]
                    if mode == 's_channel':                    
                        ch_idx = idxs[i]
                        mean = act[0, ch_idx].mean()
                        std  = act[0, ch_idx].std()
                        obj = mean / (std + eps)                          
                    elif mode == 'm_ch_sim':
                        obj = act[0, :].mean()
                    elif mode == 'm_ch_sim_norm':
                        act = F.normalize(act)
                        obj = act[0, :].mean()
                    elif mode == 'm_ch_norm':
                        norm = F.normalize(act)
                        mean = norm[0, :].mean()
                        std  = act[0, :].std()
                        obj = mean / (std + eps)
                    elif mode == 'm_ch_mean':            
                        mean = act[0, :].mean()
                        std  = act[0, :].std()
                        obj = mean / (std + eps)
                    elif mode == 's_neuron':                    
                        rel_y, rel_x = xy_list[i] if xy_list else (0.5, 0.5)
                        H_act, W_act = act.shape[2:]
                        y = int(round(rel_y * (H_act - 1)))
                        x = int(round(rel_x * (W_act - 1)))                   
                        ch_idx = idxs[i]
                        norm = F.normalize(act)
                        mean = norm[0, ch_idx, y, x].mean()
                        obj = mean / (act[0, ch_idx].std() + eps)                
                    else:
                        raise ValueError("Unknown '--mode' selected!")
                    
                (-obj).backward()
                g = tile.grad
                grad[..., y0:y1, x0:x1] += g

        yx_glo = tuple(round(x, 2) for x in yx_glo) if args.mode in ('s_neuron') else None
        grad = torch.roll(grad, (-shift_y, -shift_x), (2, 3))
        return grad / (grad.std() + 1e-8), ch_idx_glo, yx_glo


    # ------------- DeepDream Main Loop --------------------------------------------
    def deepdream_for_idx_multilayer(
            img: torch.Tensor,
            layers: List[str],
            idxs: List[int],
            lr: float,
            tag: str = '',
            xy_list: List[Tuple[int, int]] = None
    ):

        base_h, base_w = img.shape[-2:]
        octave_shapes = [(int(base_h * (args.octave_scale ** n)),
                          int(base_w * (args.octave_scale ** n)))
                         for n in args.octaves]

        work_img = img.clone()
        #ch_idx_glo yx_glo
        for o_idx, (oh, ow) in enumerate(octave_shapes):
            work_img = torch.nn.functional.interpolate(work_img,
                                                       size=(oh, ow),
                                                       mode='bilinear',
                                                       align_corners=False)
            print(f'Octave {args.octaves[o_idx]} – {oh}×{ow}')
            for step in tqdm(range(args.iters), leave=False):
                work_img.requires_grad_(True)
                grad, ch_idx_glo, yx_glo = tiled_gradient_multilayer(
                    work_img,
                    layers,
                    input_dim,
                    args.mode,
                    idxs=idxs,
                    xy_list=xy_list
                )
                work_img = (work_img + lr * grad).detach().clamp(-3, 3)
                
                if args.save_steps:
                    if step % 20 == 0:
                        to_pil(work_img).save(
                            f'rn_deepdream/{args.mode}'
                            f'_{tag}_{step}_{ch_idx_glo}_oct{args.octaves[o_idx]}.png'
                        )

            if xy_list is not None:
                to_pil(work_img).save(
                    f'rn_deepdream/{args.mode}_{args.layers}'
                    f'_{tag}_{ch_idx_glo}_{yx_glo}_oct{args.octaves[o_idx]}.png'
                )
            else:
                to_pil(work_img).save(
                    f'rn_deepdream/{args.mode}_{args.layers}'
                    f'_{tag}_{ch_idx_glo}_oct{args.octaves[o_idx]}.png'
                )
        # Final Image (upscaled)
        work_img = torch.nn.functional.interpolate(work_img,
                                                   size=(base_h*2, base_w*2),
                                                   mode='bilinear',
                                                   align_corners=False)


        if xy_list is not None:
            final_path = (f'rn_deepdream/{args.mode}_{args.layers}'
                          f'_{ch_idx_glo}_{yx_glo}_{tag}_final.png')
        else:
            final_path = (f'rn_deepdream/{args.mode}_{args.layers}'
                          f'_{ch_idx_glo}_{tag}_final.png')
        
        to_pil(work_img).save(final_path)
        print(f'Saved: {final_path}')


    # ------------- Paths / Device -------------
    os.makedirs('rn_deepdream', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------- Load CLIP ---------------
    model, preprocess = clip.load(args.model, device=device)
    model.eval().float()
    #global input_dim, num_layers
    input_dim = get_clip_dim(preprocess)
    num_layers = len(target_layers)

    # --------------- Load Image ---------------
    orig_img = Image.open(args.img).convert('RGB')
    inp_img = preprocess(orig_img).unsqueeze(0).to(device)
    img_name = os.path.splitext(os.path.basename(args.img))[0]
    embed_path = f"txtembeds/{img_name}_emb.pt"


    # -------- PRE: Gradient Ascent for Neuron mode --------

    if args.mode == 's_neuron':
        # Gradient Ascent on Text Embeddings for GradCAM
        dump_and_load = False
        if os.path.exists(embed_path) and not args.no_reload_ga:
            print(f"Using cached embedding: {embed_path}")
            text_embeds = torch.load(embed_path).to(device, dtype=model.dtype)
            model_text_dim = get_model_text_embed_dim(model)
            if text_embeds.shape[1] != model_text_dim:
                print(Fore.RED + Style.BRIGHT +
                    f"WARNING: Cached embedding shape {text_embeds.shape[1]} does not match model text dim {model_text_dim}. "  + Fore.RESET
                )
                dump_and_load = True
        else:
            print(f"Running gradient ascent for {img_name}")
            run_gradient_ascent(model, preprocess, args.img)
            text_embeds = torch.load(embed_path).to(device, dtype=model.dtype)
        
        if dump_and_load:
            print(f"Skipping reload and running gradient ascent for {img_name}")
            run_gradient_ascent(model, preprocess, args.img)
            text_embeds = torch.load(embed_path).to(device, dtype=model.dtype)
        
        model = model.float()

    # ----------- DeepDream visualisations for all requested layers ----------

    # -------------------------  C H A N N E L  ------------------------------
    if args.mode in ('s_channel', 'm_ch_norm', 'm_ch_mean', 'm_ch_sim', 'm_ch_sim_norm'):
        sal_per_layer   = [get_deepdream_channels_contrastive(model, inp_img, ln, args.k)
                           for ln in target_layers]
        scale_per_layer = [scale_salience(s) for s in sal_per_layer]

        for i in range(args.k):
            idxs   = [sal_per_layer[j][i][0]   for j in range(num_layers)]      # idx per layer
            scales = [scale_per_layer[j][i]    for j in range(num_layers)]
            lr_i   = (0.5 * args.lr * max(scales)) if args.lr_by_acts else args.lr

            print(Fore.YELLOW + Style.BRIGHT +
                  f"\nUsing LR {lr_i:.4g} for top-{i}" + Fore.RESET)

            deepdream_for_idx_multilayer(
                inp_img,
                target_layers,
                idxs,
                lr=lr_i,
                tag=f'top{i}'
            )

    # -------------------------  N E U R O N  ---------------------------
    elif args.mode in ('s_neuron'):
        k = text_embeds.shape[0]

        sal_per_layer   = [get_deepdream_channels_contrastive(model, inp_img, ln, k)
                           for ln in target_layers]
        scale_per_layer = [scale_salience(s) for s in sal_per_layer]

        clip_dim = get_clip_dim(preprocess)
        inp_img_sq = ensure_clip_square(inp_img, clip_dim)

        sal_locs_layer  = [do_gradcam(model, inp_img_sq, text_embeds, layer=ln)
                           for ln in target_layers]

        for i in range(k):
            idxs    = [sal_per_layer[j][i][0] for j in range(num_layers)]      # idx per layer
            xy_list = [sal_locs_layer[j][i]   for j in range(num_layers)]      # yx of neuron
            scales  = [scale_per_layer[j][i]  for j in range(num_layers)]
            lr_i    = (0.5 * args.lr * max(scales)) if args.lr_by_acts else args.lr

            print(Fore.YELLOW + Style.BRIGHT +
                  f"\nUsing LR {lr_i:.4g} for top-{i}" + Fore.RESET)

            deepdream_for_idx_multilayer(
                inp_img,
                target_layers,
                idxs,
                lr=lr_i,
                tag=f'top{i}',
                xy_list=xy_list
            )

    # -------------------------  F A L L B A C K  -------------------------------
    else:  # No saliency-ranking top-k; single vis
        print(Fore.YELLOW + Style.BRIGHT +
              f"\nUsing LR {args.lr}" + Fore.RESET)

        deepdream_for_idx_multilayer(
            inp_img,
            target_layers,
            idxs=[0] * num_layers, # dummy channel index per layer
            lr=args.lr,
            tag='mean_all'
        )

if __name__ == "__main__":
    main()