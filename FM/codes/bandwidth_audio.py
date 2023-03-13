import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

# Load WAV file
sample_rate, signal = wavfile.read('Sound.wav')

# Perform FFT on signal
fft1 = np.fft.fft(signal)
mag=np.abs(fft1)

# Calculate power spectral density
psd1 = np.abs(fft1)**2

# Calculate frequency range
freqs1 = np.fft.fftfreq(len(psd1), 1/sample_rate)

# Find frequency range with significant power
mask1 = psd1 > 0.1*np.max(psd1)
freq_range1 = freqs1[mask1]
bandwidth1 = max(freq_range1) - min(freq_range1)

print('Bandwidth:', bandwidth1)

plt.plot(freqs1,mag)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain (dB)')
plt.grid(True)
plt.title('Audio Signal Spectrum')
#plt.legend(loc='best')
plt.show()


