import numpy as np
from tenseur import Clip, Live, seed, drum_tensor, random_tensor, normal, bernoulli, euclidean, simple_tensor, upsample, scatter, generate_scale
from tracks import *

# region scene parameters 
CLIP = 2
ROOT = 57
SEED = 42
SCALE = generate_scale(7, root=ROOT, offset=-1/7)
seed(SEED)
live = Live(index=CLIP)
# endregion

# region Noise
noise = simple_tensor((2, 4, 16), pitch=36, duration=1)
noise[0, :, 12, 1] = 1  #velocity
noise[0, :, 12, 3] = 4  #duration
Clip(noise).track(5).speed(1).show().render(live)
# endregion

# region Sub
sub = simple_tensor([2, 4, 16], duration=4)
sub[0, ::2, :9:3, 1] = 1
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(1).show().render(live)
# endregion

# region Drums
drums = drum_tensor((16, 4, 16), duration=1, kit=5)
drums[0, :, [0, 6, 11, 13], 1] = 1  #kick
drums[2, :, [4, 12], 1] = 1  #snare
drums[4, :, :, 1] = 1
drums[..., 1] = bernoulli(prob=0.2)
drums[..., 1] *= bernoulli(prob=0.1)
drums[:,3, :, 1] *= drums[:,3, np.arange(16)%7, 1]
Clip(drums).sparsify([0, 1, 2, 3]).quantize(1).track(4).show().render(live)
# endregion

# region Piano
piano = random_tensor((3, 16, 16), prob=0.05, scale=3, dur_scale=3, range=14)
Clip(piano).track(9).humanize().linear_pitch(12/7, ROOT+1*12).project_pitch(SCALE).speed(1/2).show().render(live)
# endregion



live.stop()