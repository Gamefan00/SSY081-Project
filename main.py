
import numpy as np
import matplotlib.pyplot as plt

# Construct signal
action_pot = np.load('./data_files/action_potentials.npy')

datafr = np.load('./data_files/firing_samples.npy', allow_pickle=True)[0]

UNIT_COUNT = 8



def plot(time, signal):
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal, linewidth=0.5)

    # Label the axes
    plt.xlabel('Time [s]')
    plt.ylabel('[A.U.]')

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.show()


# create 8x200000 matrix where each row corresponds to a unit and each column to a time step
ap_trains  = np.zeros((UNIT_COUNT, 200000), dtype=float)

# fill the matrix with the action potentials
for t in range(datafr.size):
    for i in (datafr[t]):
        i = i[0]
        ap_trains[t, i:i+100] += np.array(action_pot[t])

# combine the action potentials
combined = np.sum(ap_trains, axis=0)

# Plot for question 1
if False:
    time = np.arange(len(ap_trains[0])) / 10000
    plot(time, ap_trains[0])
    plot(time[100000:105000], ap_trains[0,100000:105000])
    plot(time[100000:105000], combined[100000:105000])



