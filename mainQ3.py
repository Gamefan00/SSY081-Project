
import numpy as np
import matplotlib.pyplot as plt
from math import pi

SAMPLING_FRQ = 1024

# Load data
signal = np.load('./data_files/f.npy').reshape(-1)

# sample a sinusodal wave at 1024 Hz
# wave speed = 50 hz, amplitude = 0.3
interference = 0.3 * np.sin(2 * np.pi * 50 * np.arange(0, 1, 1/SAMPLING_FRQ))

# add the interference to the signal
signal_interf = signal + interference

time_vec = np.arange(len(signal)) / SAMPLING_FRQ


plt.figure(figsize=(12, 4))
plt.plot(time_vec, signal_interf)
plt.plot(time_vec, signal)
plt.xlabel('Time [s]')
plt.show()


N0 = len(signal)            # number of samples (1024)
T0 = N0 / SAMPLING_FRQ      # total time (1024 samples, at 1024 hz => 1s)

signal_dft = np.fft.fft(signal)
signal_interf_dft = np.fft.fft(signal_interf)

amplitude = np.abs(signal_dft)
amplitude_interf = np.abs(signal_interf_dft)

# phase = np.arctan2(signal_dft.imag, signal_dft.real)
# phase_interf = np.arctan2(signal_interf_dft.imag, signal_interf_dft.real)

k = np.arange(0, N0, step=1)
f_axis = k / T0            # get the axis in Hz
w_axis = 2 * pi * k / T0   # get the axis in rad/s

# plot absolute values of the DFT
plt.figure(figsize=(12, 4))
plt.plot(f_axis[0:int(N0/2)], amplitude_interf[0:int(N0/2)], linewidth=0.5)
plt.plot(f_axis[0:int(N0/2)], amplitude[0:int(N0/2)], linewidth=0.5)
plt.xlabel('Frequency [Hz]')
plt.ylabel('|F| [a.u.]')
plt.xlim(0, 100)
plt.show()