
import numpy as np
import matplotlib.pyplot as plt

# Construct signal
action_pot = np.load('./data_files/action_potentials.npy')

datafr = np.load('./data_files/firing_samples.npy', allow_pickle=True)[0]

UNIT_COUNT = 8
SAMPLING_FRQ = 10000


def plot(time, *signals):
    plt.figure(figsize=(10, 4))

    for signal in signals:
        plt.plot(time, signal, linewidth=0.5)

    # Label the axes
    plt.xlabel('Time [s]')
    plt.ylabel('[A.U.]')

    # Show grid
    plt.grid(True)

    # Display the plot
    plt.show()


# Question 1
if False:
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
    time = np.arange(len(ap_trains[0])) / SAMPLING_FRQ
    plot(time, ap_trains[0])
    plot(time[100000:105000], ap_trains[0,100000:105000])
    plot(time[100000:105000], combined[100000:105000])



# Question 2
if True:
    hanning_window = np.hanning(1*SAMPLING_FRQ)

    # Create binary firing train
    q2_binary = np.zeros((UNIT_COUNT, 200000), dtype=int)
    q2_filtered = np.zeros((UNIT_COUNT, 200000), dtype=float)
    for t in range(datafr.size):
        for i in (datafr[t]):
            i = i[0]
            q2_binary[t, i] = 1
        
        # convolve resulting binary firing train with hanning
        q2_filtered[t] = np.convolve(q2_binary[t], hanning_window, mode='same')

    # Plot for question 2
    time = np.arange(len(q2_filtered[0])) / SAMPLING_FRQ
    
    # C) plot all signals in one figure
    plot(time, *q2_filtered)
    
    # D) Plot fourth binary vector and corresponding filtered signal
    plot(time, q2_binary[3], q2_filtered[3])
    
    # E) Plot seventh binary vector and corresponding filtered signal
    plot(time, q2_binary[6], q2_filtered[6])

