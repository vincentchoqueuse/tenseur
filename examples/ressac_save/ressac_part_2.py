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

# region Drums
drums = drum_tensor((16, 4, 16), duration=1, kit=6)
drums[0, :, [0, 3, 6, 13], 1] = 1
drums[2, :, [4, 12], 1] = 1
drums[3, [0, 2], 7:13:2, 1] = 1
drums[3, [1], 7:13:3, 1] = 1
drums[3, [3], 5:13:2, 1] = 1
drums[4, :, :, 1] = 1
drums[4, :, 0, 1] = 0
drums[14, :, :, 1] = euclidean(7, 16)
#drums[:, :, :, 1] += 0.1*bernoulli((16, 4, 16), 0.4)
Clip(drums).rgate(0.99).sparsify([0, 1, 2, 3, 8]).speed(1).quantize(1).track(4).show().render(live)
# endregion

# region Sub
sub = simple_tensor([2, 2, 16], duration=16)
sub[0, 1, 8, 1] = 1
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(1).show().render(live)
# endregion

