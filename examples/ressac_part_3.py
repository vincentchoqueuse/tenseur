import numpy as np
from tenseur import Clip, Live, seed, random_tensor, drum_tensor, normal, bernoulli, euclidean, simple_tensor, upsample, scatter, generate_scale
from tracks import *

# region scene parameters 
CLIP = 3
ROOT = 52
SEED = 42
SCALE = generate_scale(7, root=ROOT, offset=-1/7)
seed(SEED)

live = Live(index=CLIP)
# endregion

speed = 1/4


# region Drums
drums = drum_tensor((16, 4, 16), duration=1, kit=3)
drums[0, :, :9:3, 1] = 1  #kick
drums[2, :, 12, 1] = 1  #snare
drums[4, :, :, 1] = 1
#drums[..., 1] = bernoulli(prob=0.3)
drums[..., 1] *= bernoulli(prob=0.2)
Clip(drums).sparsify([0, 1, 2, 3, 4]).quantize(1).track(4).show().render(live)
# endregion


# region Sub
sub = simple_tensor([2, 4, 16], duration=4)
sub[0, 3, 12, 1] = 1
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(1).show().render(live)
# endregion

# region Reese
V = [0, 0, -2, -1]  #pitch
reese = simple_tensor((4, 4, 16), duration=36)
reese[0, :, 0, 0] = V
reese[0, [0, 2, 3], 0, 1] = 1
Clip(reese).track(7).linear_pitch(12/7, ROOT+0*12).project_pitch(SCALE).speed(1/2).show().render(live)
# endregion

# region Piano
piano = random_tensor((3, 16, 16), prob=0.1, scale=3, dur_scale=3, range=14)
Clip(piano).track(9).humanize().linear_pitch(12/7, ROOT+1*12).project_pitch(SCALE).speed(1/2).show().render(live)
# endregion


live.stop()