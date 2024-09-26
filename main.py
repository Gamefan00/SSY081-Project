
import numpy as np

# Hello there
print("Hello There!")

action_pot = np.load('./data_files/action_potentials.npy')

datafr = np.load('./data_files/firing_samples.npy', allow_pickle=True)[0]

bin_fire  = np.zeros((8, 200000), dtype=float)


print(action_pot[0].shape)
print(bin_fire[1, 1:1+100].shape)

#exit()
for t in range(datafr.size):
    for i in (datafr[t]):
        i = i[0]
        bin_fire[t, i:i+100] += np.array(action_pot[t])

print(bin_fire[7, 1200:1500])
print(datafr.shape)
print(action_pot.shape)
