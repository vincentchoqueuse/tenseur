import numpy as np
from tenseur import Clip, Live, crossfade, seed, random_tensor, rdrum_tensor, drum_tensor, bernoulli, normal, euclidean, simple_tensor, upsample, scatter, generate_scale
from tracks import *

# region scene parameters 
CLIP = 1
ROOT = 52
SEED = 42
SCALE = generate_scale(7, root=ROOT, offset=-1/7)
seed(SEED)
live = Live(index=CLIP)
# endregion

# region Sampler
sounds = {"BEEP1": 48, "BEEP2": 49}
sampler = simple_tensor((2, 4, 16), pitch=sounds["BEEP1"])
sampler[0, :, 0, 1] = 1
sampler[1, 3, 8, 0] += 1
sampler[1, 3, 8, 1] = 1
Clip(sampler).track(10).speed(1).show().render(live)
# endregion

# region Noise
noise = simple_tensor((2, 4, 16), pitch=36, duration=1)
noise[0, :, 12, 1] = 1  #velocity
noise[0, :, 12, 3] = 4  #duration
Clip(noise).track(5).speed(1).show().render(live)
# endregion

# region Sub
sub = simple_tensor([2, 4, 16], duration=4)
sub[0, ::2, 12, 1] = 1
Clip(sub).track(6).linear_pitch(12/7, ROOT-2*12).project_pitch(SCALE).speed(1).show().render(live)
# endregion

# region Drums
drums = drum_tensor((16, 4, 16), duration=1, kit=3)
drums[0, :, :9:3, 1] = 1  #kick
drums[2, :, 12, 1] = 1  #snare
drums[4, :, :, 1] = 1
#drums[..., 1] = bernoulli(prob=0.3)
drums[..., 1] *= bernoulli(prob=0.02)
Clip(drums).sparsify([0, 1, 2, 3, 4]).quantize(1).track(4).show().render(live)
# endregion

# region Key
key = simple_tensor((4, 2, 16))
key[0, :, :, 0] = np.arange(16)%11%5  #pitch
key[0, :, :, 1] = euclidean(15, 16)  #velocity
key[1, :, :, 0] = 14 - key[0, :, :, 0] 
key[1, :, :, 1] = 0.4*euclidean(5, 16, rotation=2)  #velocity
Clip(key).track(8).linear_pitch(12/7, ROOT).project_pitch(SCALE).show().speed(1).render(live)
# endregion

live.stop()



