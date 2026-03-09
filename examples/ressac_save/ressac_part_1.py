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
key[0, :, :, 1] = euclidean(15, 16)
key[1, :, :, 0] = 14 - key[0, :, :, 0]
key[1, :, :, 1] = 0.7*euclidean(9, 16, rotation=1)
Clip(key).track(7).linear_pitch(12/7, ROOT).project_pitch(SCALE).show().speed(1).render(live)
# endregion

# region Drums
drums = drum_tensor((16, 4, 16), duration=1, kit=5)
drums[0, :, [0, 9], 1] = 1
drums[0, 3, [7, 13], 1] = 1
drums[1, :, [3, 6, 11], 1] = 0.4
drums[2, :, [4, 12], 1] = 1
drums[4, :, :, 1] = 1
drums[:, 3, :, 2] += 3.5*normal((16, 16))
drums[4:, 1, :, 1] += 0.8*bernoulli((12, 16), 0.1)
drums[4:, 3, :, 1] += 0.8*bernoulli((12, 16), 0.5)
#drums[0, ..., 1] = 1
Clip(drums).rgate(1).sparsify([0, 1, 2, 3]).speed(1).quantize(1).track(4).show().render(live)
# endregion

# region Noise
noise = simple_tensor((2, 4, 16), pitch=42)
noise[0, [1, 3], 0, 1] = 1
noise[0, 3, 0, 3] = 4
Clip(noise).track(5).speed(1).render(live)
# endregion

# region Sub
sub = simple_tensor((2, 4, 16), duration=6)
#sub[0, [1, 3], 10, 1] = 1
sub[0, :, 0:12:3, 1] = 1
sub[0, :, 6, 0] = 3
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(1).show().render(live)
# endregion


# region Sampler
sounds = {"BEEP1": 48, "BEEP2": 49}
sampler = simple_tensor((2, 4, 16), pitch=sounds["BEEP2"])
sampler[0, :, 4, 1] = 1
Clip(sampler).track(8).speed(1).show().render(live)
# endregion

live.stop()


