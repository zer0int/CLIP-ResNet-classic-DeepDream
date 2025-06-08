# 🌌 Classic DeepDream with CLIP ResNet 🤖💭💫

### Recommended Quickstart options:

- Single, most salient channel [0]:
 ```
python clip_deepdream_rn.py --img img/lab.jpg --layers layer3 --mode s_channel --lr 0.008
 ```
- Single, most salient channel [1], filtered out 'generalizing' activations:
 ```
python clip_deepdream_rn.py --img img/lab.jpg --layers layer3 --mode s_channel --filter_acts --lr 0.008
 ```
- 🌟 All channels of a layer3 [2], but nudged by mean/std and norm:
 ```
python clip_deepdream_rn.py --img img/lab.jpg --layers layer3 --mode m_ch_norm --lr 0.005
 ```
- 💫 Neuron mode, two layers [3]: Noisier image (you can try: decrease to --lr 0.01):
 ```
python clip_deepdream_rn.py --img img/lab.jpg --layers layer2 layer3 --mode s_neuron --lr 0.03
 ```
![ResNet-DeepDream](https://github.com/user-attachments/assets/e81d6b1d-821e-4b34-a54b-cccdb7e3d73e)
----------

- Single ("s_") most salient channel:
- Optional: filter generalizing 'salient to everything' activations
 ```
--mode s_channel --filter_acts
 ```

- ALL channels ("m_" is for "multi") of target layer:
- For fun, also select a different model. Default is RN50x4.
 ```
--mode m_ch_norm --model RN101
 ```

- Single most salient neuron:
- Uses Gradient Ascent on Text + GradCAM for attention-informed selection
- Batch size is 10 by default; if OOM for large models, reduce it:
 ```
--mode s_neuron --batch_size_ga 6 --k 6
 ```

- Default for octaves is -3 to +3 and scale 1.5; for lower resolution (much faster), try:
 ```
--octaves -2 -1 0 1 2 --octave_scale 1.3
 ```

- For ALL available arguments, do:
 ```
python clip_deepdream_rn.py --help
 ```

- CLIP 'seeing tiger' and 'making tiger' (s_neuron mode):

![resnet-ga-neuron](https://github.com/user-attachments/assets/b3db2eb8-ee05-41df-a0e7-c382c5a1ac4b)

- Channel mode; use `--save_steps` to save every 20th step:

https://github.com/user-attachments/assets/b4ad8ce1-bc04-4b08-9a63-25aadcd189b5

----------

### Novel DeepDream with CLIP ViT

- Very different technique, but good for comparison.
- Use a square image for ViT input
- See `clip_deepdream_vit.py --help` for all args

 ```
python clip_deepdream_vit.py --img img/lab.png
 ```
![vit-dream](https://github.com/user-attachments/assets/38310c26-6dc6-4d3d-bb89-bb5c6923e8c6)

- See also (for more ViT): [zer0int/CLIP-DeepDream](https://github.com/zer0int/CLIP-DeepDream)

----------

- Image source: 'dog: labrador' from original DeepDream, [www.tensorflow.org/tutorials/generative/deepdream](https://www.tensorflow.org/tutorials/generative/deepdream)
- Rats: [self] (quality: very old!)
