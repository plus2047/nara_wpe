#%%
root_path = "../"

import sys
sys.path = [root_path] + sys.path

import nara_wpe.wpe
import nara_wpe.utils
import numpy as np
from scipy.io import wavfile

#%%
test_file = root_path + "data_local/ping-pang.wav"
output_origin = root_path + "data_local/ping-pang-origin.wav"
output_filted = root_path + "data_local/ping-pang-filted.wav"
data_length = 1.5 #s
chan_num = 1

stft_size = 2048
stft_shift = 512
wpe_delay = 3
wpe_iter = 5
wpe_taps = 10

#%%
fs, X = wavfile.read(test_file)
X = X[:int(fs * data_length), :chan_n]
stft_opts = dict(size=stft_size, shift=stft_shift)
Y = nara_wpe.utils.stft(X.T, **stft_opts).transpose(2, 0, 1)

Z, G = nara_wpe.wpe.wpe(
    Y,
    taps=wpe_taps,
    delay=wpe_delay,
    iterations=wpe_iter,
    statistics_mode='full',
    return_filter=True
)
Z = Z.transpose(1, 2, 0)

z = nara_wpe.utils.istft(Z, **stft_opts).T
origin_len = X.shape[0]
filted_data = z[:origin_len, :].astype("float32")
origin_data = X.astype("float32")
maxVal = max(np.max(np.abs(X)), np.max(np.abs(filted_data)))
filted_data /= maxVal
origin_data /= maxVal

# wavfile.write(output_origin, fs, origin_data)
# wavfile.write(output_filted, fs, filted_data)

#%%
def inv_wpe(X, G, taps, delay, Y_init=None):
    # Z: (chan_num, L, stft_half_size)
    # G: (stft_half_size, taps * chan_num, chan_num)
    X = X.transpose(2, 0, 1)
    Y = np.zeros_like(X)
    stft_length, chan_num, L = X.shape
    Y_init_length = 0 if Y_init is None else Y_init.shape[-1]
    for t in range(L):
        Xt = X[:, :, t][:, :, None]
        Yt = np.copy(Xt)
        for tau in range(delay, delay + taps):
            if tau >= t:
                break
            _tau = tau - delay
            Gt = G[:, chan_num * _tau:chan_num * (_tau + 1), :]
            Y_ref = Y_init if t - tau < Y_init_length else Y
            Y_tilde_t = Y_ref[:, :, t - tau][:, :, None]
            Yt += np.matmul(nara_wpe.wpe.hermite(Gt), Y_tilde_t)
        # if t < Y_init_length:
        #     print(np.linalg.norm(Y_init[:, :, t] - Y[:, :, t]))
        Y[:, :, t] = Yt[:, :, 0]
    return Y

# %%
rec_Y = inv_wpe(Z, G, wpe_taps, wpe_delay, Y_init=Y[:, 
:, :40])
diff = np.linalg.norm(rec_Y - Y, axis=(0, 1))
print(diff)
