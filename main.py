
import numpy as np
import matplotlib.pyplot as plt

# Construct signal
action_pot = np.load('./data_files/action_potentials.npy')

datafr = np.load('./data_files/firing_samples.npy', allow_pickle=True)[0]

UNIT_COUNT = 8
SAMPLING_FRQ = 10000
SAMPLE_COUNT = 200000


def plot(time, *signals):
    plt.figure(figsize=(10, 4))

    for signal in signals:
        plt.plot(time, signal, linewidth=0.5)

    # Label the axes
    plt.xlabel('Time [s]')
    plt.ylabel('[A.U.]')

    plt.grid(True)

    # Display the plot
    plt.show()


# Create binary firing train based on indicies in datafr
binary_trains = np.zeros((UNIT_COUNT, SAMPLE_COUNT), dtype=int)
for i, indicies in enumerate(datafr):
    binary_trains[i, indicies] = 1

# Create time vector (for plotting)
time = np.arange(SAMPLE_COUNT) / SAMPLING_FRQ


# Question 1 ---------------------------------------------------------------
if True:
    # create 8x200000 matrix where each row corresponds to a unit and each column to a time step
    # data created by convolving the binary firing train with the action potentials
    ap_trains = np.array([
        np.convolve(binary_trains[i], action_pot[i], mode='same') 
        for i in range(UNIT_COUNT)
    ])

    # alternative method:
    # fill the matrix with the action potentials
    # for t in range(UNIT_COUNT):
        # for i in (datafr[t]):
            # i = i[0]
            # ap_trains[t, i:i+100] += np.array(action_pot[t])

    # combine the action potentials
    combined = np.sum(ap_trains, axis=0)

    # Plot for question 1
    plot(time, ap_trains[0])
    plot(time[100000:105000], ap_trains[0,100000:105000])
    plot(time[100000:105000], combined[100000:105000])



# Question 2 ---------------------------------------------------------------
if True:
    hanning_window = np.hanning(1*SAMPLING_FRQ)

    # Create binary firing train, then convolve with hanning window
    # q2_binary = np.zeros((UNIT_COUNT, SAMPLE_COUNT), dtype=int)
    # q2_filtered = np.zeros((UNIT_COUNT, SAMPLE_COUNT), dtype=float)

    q2_filtered = np.array([
        np.convolve(binary_trains[i], hanning_window, mode='same') 
        for i in range(UNIT_COUNT)
    ])

    # equivilant to:
    # for t in range(UNIT_COUNT):
        # for i in (datafr[t]):
            # i = i[0]
            # q2_binary[t, i] = 1
        
        # convolve resulting binary firing train with hanning
        # q2_filtered[t] = np.convolve(q2_binary[t], hanning_window, mode='same')
    
    # C) plot all signals in one figure
    plot(time, *q2_filtered)
    
    # D) Plot fourth binary vector and corresponding filtered signal
    plot(time, binary_trains[3], q2_filtered[3])
    
    # E) Plot seventh binary vector and corresponding filtered signal
    plot(time, binary_trains[6], q2_filtered[6])

