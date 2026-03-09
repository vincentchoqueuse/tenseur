import numpy as np
from tenseur import Clip, Live, seed, drum_tensor, normal, bernoulli, euclidean, simple_tensor, upsample, scatter, generate_scale
from tracks import *

# region scene parameters 
CLIP = 2
ROOT = 57
SEED = 42
SCALE = generate_scale(7, root=ROOT, offset=-1/7)
seed(SEED)
live = Live(index=CLIP)
# endregion

# region Sub
sub = simple_tensor([2, 4, 16], duration=4)
sub[0, :, :6:3, 1] = 1
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(1).show().render(live)
# endregion

# region Sampler
sounds = {"BEEP1": 48, "BEEP2": 49}
sampler = simple_tensor((2, 4, 16), pitch=sounds["BEEP1"])
sampler[0, :, 12, 1] = 1
Clip(sampler).track(8).speed(1).show().render(live)
# endregion

# region Drums
drums = drum_tensor((16, 4, 16), duration=1, kit=5)
drums[0, :, [0, 3, 6 ,11, 13], 1] = 1
drums[2, :, [4, 12], 1] = 1
drums[3, :3, 5::3, 1] = 0.4
drums[3, 3, 5::2, 1] = 0.4
drums[4, :, :, 1] = 1
drums[:, 3, :, 2] += 2*normal((16, 16))
drums[8, :, :, 1] = euclidean(13, 16)
drums[6:, 3, :, 1] = bernoulli((10, 16), 0.3)
Clip(drums).rgate(0.99).sparsify([0, 1, 2, 3]).quantize(1).speed(1).track(4).show().render(live)
# endregion