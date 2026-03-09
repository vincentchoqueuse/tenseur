import numpy as np
from tenseur import Clip, Live, seed, drum_tensor, normal, bernoulli, euclidean, simple_tensor, upsample, scatter, generate_scale
from tracks import *

# region scene parameters 
CLIP = 2
ROOT = 52
SEED = 42
SCALE = generate_scale(7, root=ROOT, offset=-1/7)
seed(SEED)
live = Live(index=CLIP)
# endregion

speed = 1/4

M = np.array([
    [0, -1, -1, -1],
    [2, 2, 1, 0],
    [4, 4, 4, 3],
    ]
)
M[0, :] -= 7
M[1, :] += 7

# region Synth2
synth2 = simple_tensor((3, 4, 16), duration=16)
synth2[..., 0, 1] = 1
synth2[..., 0, 0] = M
Clip(synth2).track(10).linear_pitch(12/7, ROOT+12).project_pitch(SCALE).show().speed(speed).render(live)
# endregion

