import sys
sys.path = ["."] + sys.path

import nara_wpe.wpe
import nara_wpe.utils
import numpy as np
from scipy.io import wavfile


test_file = "data_local/ping-pang.wav"
output_origin = "data_local/ping-pang-origin.wav"
output_filted = "data_local/ping-pang-filted.wav"
data_length = 3 #s

stft_size = 2048
stft_shift = 512
wpe_delay = 3
wpe_iter = 5
wpe_taps = 10

fs, X = wavfile.read(test_file)
X = X[:int(fs * data_length), :]
stft_opts = dict(size=stft_size, shift=stft_shift)
Y = nara_wpe.utils.stft(X.T, **stft_opts).transpose(2, 0, 1)
Z = nara_wpe.wpe.wpe(
    Y,
    taps=wpe_taps,
    delay=wpe_delay,
    iterations=wpe_iter,
    statistics_mode='full'
).transpose(1, 2, 0)

z = nara_wpe.utils.istft(Z, **stft_opts).T
origin_len = X.shape[0]
filted_data = z[:origin_len, :].astype("float32")


origin_data = X.astype("float32")
maxVal = max(np.max(np.abs(X)), np.max(np.abs(filted_data)))
filted_data /= maxVal
origin_data /= maxVal
wavfile.write(output_origin, fs, origin_data)
wavfile.write(output_filted, fs, filted_data)
