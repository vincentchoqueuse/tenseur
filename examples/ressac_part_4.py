import numpy as np
from tenseur import Clip, Live, seed, random_tensor, drum_tensor, normal, bernoulli, euclidean, simple_tensor, upsample, scatter, generate_scale
from tracks import *


# region scene parameters 
CLIP = 4
ROOT = 52
SEED = 42
SCALE = generate_scale(7, root=ROOT, offset=-1/7)
seed(SEED)

live = Live(index=CLIP)
# endregion

speed = 1/4


# region Reese
V = [0, 0, -2, -1]  #pitch
reese = simple_tensor((4, 4, 16), duration=36)
reese[0, :, 0, 0] = V
reese[0, [0, 2, 3], 0, 1] = 1
Clip(reese).track(7).linear_pitch(12/7, ROOT+0*12).project_pitch(SCALE).speed(1/2).show().render(live)
# endregion
live.stop()