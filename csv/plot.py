import numpy as np
import matplotlib.pyplot as plt
import csv

x = []

with open("./csv/rope_load.csv", "r") as csvfile:
    plots = csv.reader(csvfile, delimiter=",")

    for row in plots:
        x.append(row[0])
last200_x = x[:-200]
plt.plot(np.arange(len(last200_x)), last200_x, label="rope_load")
plt.show()
