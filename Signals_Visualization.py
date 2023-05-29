#This code will allow you to visualize an example of a VGRF signal from a left foot of both, control and parkonsinian patient.

import matplotlib.pyplot as plt
plt.style.use("ggplot")
# read in the data from the first text file
time1 = []
signal1 = []
with open("/content/GaCo03_01.txt") as f:
    for line in f:
        values = line.split()
        # only read in values from the first 10 seconds
        if float(values[0]) <= 10.0:
            time1.append(float(values[0]))
            signal1.append(float(values[17]))

# read in the data from the second text file
time2 = []
signal2 = []
with open("/content/GaPt03_01.txt") as f:
    for line in f:
        values = line.split()
        # only read in values from the first 10 seconds
        if float(values[0]) <= 10.0:
            time2.append(float(values[0]))
            signal2.append(float(values[17]))

# plot the data using matplotlib
plt.plot(time1, signal1, color="blue", label="Healthy Gait")
plt.plot(time2, signal2, color="red", label="PD Gait")
plt.xlabel("Time (s)")
plt.ylabel("Total Force Under the Left Foot")
plt.legend(loc="upper left")
plt.show()
