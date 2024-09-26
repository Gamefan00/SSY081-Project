
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

plt.figure(figsize=(10, 4))
plt.plot(time, ap_trains[0], linewidth=0.5)

# Label the axes
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (A.U.)')

# Set title for the plot
plt.title('Signal over Time')

# Show grid
plt.grid(True)

# Display the plot
plt.show()
