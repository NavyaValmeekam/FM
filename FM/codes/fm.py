from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal

# Load WAV file
sample_rate, audio = wavfile.read('Sound.wav')

# Perform FFT on signal
fft = np.fft.fft(audio)

# Calculate power spectral density
psd = np.abs(fft)**2

# Calculate frequency range
freqs = np.fft.fftfreq(len(psd), 1/sample_rate)

# Plotting input spectrum using power vs freq
plt.figure(1)
plt.plot(freqs, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.title('Input Signal')
plt.show()

# Find frequency range with significant power
mask = psd > 0.1*np.max(psd)
freq_range = freqs[mask]
bandwidth = max(freq_range) - min(freq_range)

print('Bandwidth:', bandwidth, 'Hz')

# Frequency modulation
fc = 100e6  # carrier frequency

N = len(audio)
dt = 1e-3/N             # time span 1 msec
t = np.arange(N)
t1 = t*dt

audio = 127*(audio.astype(np.int16)/ np.power(2,15))
ym = audio#[:,1] # Audio data
kf = 25            # freq sensitivity
cumsum = np.cumsum(ym)  # Discrete summation

c = np.cos(2*np.pi*fc*t1)
y_fm = np.cos(2*np.pi*fc*t1 + kf*cumsum*(1/sample_rate))

# Compute PSD of FM signal
fm_fft = np.fft.fft(y_fm)
fm_psd = np.abs(fm_fft)**2
fm_freqs = np.fft.fftfreq(len(fm_psd), 1/sample_rate)
fm_freqs1 = np.fft.fftfreq(len(fm_psd), d=dt)
# Find frequency range with significant power of FM signal
threshold = 0.1*np.max(fm_psd)
fm_mask = fm_psd > threshold
fm_freq_range = fm_freqs[fm_mask]
fm_bandwidth = max(fm_freq_range) - min(fm_freq_range)
print('FM Bandwidth:', fm_bandwidth, 'Hz')

# Plot PSD of FM signal
plt.figure(2)
plt.plot(fm_freqs1, fm_psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)
plt.title('FM Signal ')
plt.legend(loc='best')
plt.show()

#multiplication with cos(2*pi*fc*t)
y_mul=y_fm*np.cos(2*np.pi*fc*t1)

fmul_fft=np.fft.fft(y_mul)
f_2=np.fft.fftfreq(len(y_mul), 1/sample_rate)

plt.figure(3)
plt.plot(f_2,abs(fmul_fft))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('local oscillator')
plt.show()

# Low-pass filter
cutoff = 3e3 # 10 kHz cutoff frequency

b, a = signal.butter(6, cutoff/(sample_rate/2), 'low') # th order Butterworth filter
z = signal.filtfilt(b, a, y_mul)
z1 = np.fft.fft(z)
z_f=np.fft.fftfreq(len(z),1/sample_rate)
plt.plot(z_f, abs(z1))
plt.title('Filtered FM signal')
plt.show()

y5 = 20*np.arccos(z)    # cos inverse of LPF output


# Differentiation of cos inverse
i=0
y6 = np.zeros(len(t))

while i< len(t)-1:
      y6[i] = (y5[i+1] - y5[i]) / (t[i+1] - t[i])
      i = i+1
      
z5 = 2*np.fft.fft(y6)
f_5 = np.fft.fftfreq(len(y6), 1/sample_rate)
plt.plot(f_5,abs(z5))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.title('Final Output Signal')
plt.show()

     
wavfile.write('signal_out.wav', sample_rate, y6)

