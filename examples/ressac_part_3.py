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
    [2, 2, 1, 1],
    [4, 4, 4, 3],
    [7, 6, 6, 6]
    ]
)

M = np.array([
    [0, -1, -1, -1],
    [2, 2, 1, 1],
    [4, 4, 4, 3],
    ]
)

M[1, :] += 7

V = M[0, :]

# region Sub
sub = simple_tensor([1, 4, 16], duration=4)
sub[..., 2, 0] = V
sub[..., 2, 1] = 1
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(speed).show().render(live)
# endregion

# region Synth1
synth1 = simple_tensor((3, 4, 16), duration=8)
synth1[:, :, 0, 1] = 1
synth1[..., 0] = M[:, :, None]
Clip(synth1).track(9).linear_pitch(12/7, ROOT-0*12).project_pitch(SCALE).show().speed(speed).render(live)
# endregion

# region Synth2
synth2 = simple_tensor((3, 4, 16), duration=8)
synth2[..., 0, 1] = 1
synth2[..., 0, 0] = M
Clip(synth2).track(10).linear_pitch(12/7, ROOT+12).project_pitch(SCALE).show().speed(speed).render(live)
# endregion

# region Synth2
gate = np.arange(8)%3

synth2 = simple_tensor((3, 4, 8), duration=8)
synth2[..., :, 0] = 1
synth2[..., 0] = M[..., None]
synth2[0,: , :, 1] = (gate==0)
synth2[1,: , :, 1] = (gate==1)
synth2[2,: , :, 1] = (gate==2)
Clip(synth2).track(11).linear_pitch(12/7, ROOT+0*12).project_pitch(SCALE).show().speed(speed).render(live)
# endregion
