
import numpy as np
import matplotlib.pyplot as plt

# Construct signal
action_pot = np.load('./data_files/action_potentials.npy')

datafr = np.load('./data_files/firing_samples.npy', allow_pickle=True)[0]

UNIT_COUNT = 8

ap_trains  = np.zeros((UNIT_COUNT, 200000), dtype=float)

for t in range(datafr.size):
    for i in (datafr[t]):
        i = i[0]
        ap_trains[t, i:i+100] += np.array(action_pot[t])


# Plot
time = np.arange(len(ap_trains[0])) / 10000

fig,axs = plt.subplots(4,2,figsize=(12,10))

for i, signal in enumerate(ap_trains):
    row = i//2
    col = i % 2
    axs[row, col].plot(time, signal, linewidth=0.5)
    axs[row, col].set_title(f'Signal {i+1}')
    axs[row, col].set_xlabel('Time (seconds)')
    axs[row, col].set_ylabel('Amplitude (A.U.)')
    axs[row, col].grid(True)

plt.tight_layout()
plt.show()
