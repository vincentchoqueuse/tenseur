import numpy as np
from tenseur import Clip, Live, seed, drum_tensor, bernoulli, normal, euclidean, simple_tensor, upsample, scatter, generate_scale
from tracks import *

# region scene parameters 
CLIP = 1
ROOT = 52
SEED = 42
SCALE = generate_scale(7, root=ROOT, offset=-1/7)
seed(SEED)
live = Live(index=CLIP)
# endregion

# region Key
key = simple_tensor((4, 2, 16))
key[0, :, :, 0] = np.arange(16)%13%7
key[0, :, :, 1] = euclidean(13, 16)
key[1, :, :, 0] = 14 - key[0, :, :, 0]
key[1, :, :, 1] = 0.6**euclidean(9, 16, rotation=1)
Clip(key).track(7).linear_pitch(12/7, ROOT).project_pitch(SCALE).show().speed(1).render(live)
# endregion

# region Drums
drums = drum_tensor((16, 4, 16), duration=1, kit=0)
drums[:, :, :, 1] = 0.2*bernoulli((16, 4, 16), 0.2)
drums[0, :, [0, 11], 1] = 1
drums[1, :, [3, 9, 13], 1] = 1
drums[2, :, [4, 12], 1] = 1
drums[3, :, 5:13:3, 1] = 1
drums[4, :, :, 1] = 1
drums[5, :, ::3, 1] = 1
drums[:, :, :, 2] += 0*normal((16, 4, 16))
drums[:, :, :, 1] = drums[:, :, np.arange(16)%13, 1]
Clip(drums).rgate(0.9).sparsify([0, 1, 2, 3, 4]).quantize(1).track(4).show().render(live)
# endregion

# region Noise
noise = simple_tensor((2, 4, 16), pitch=46)
noise[0, [1, 3], 12, 1] = 1
noise[0, 3, 12, 3] = 4
Clip(noise).track(5).speed(1).render(live)
# endregion

# region Sub
sub = simple_tensor([2, 4, 16], duration=4)
sub[0, [0, 2], :12:3, 1] = 1
sub[0, 3, 8, 1] = 1
sub[0, 3, 8, 0] = -1
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(1).show().render(live)
# endregion

live.stop()


