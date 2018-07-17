import numpy as np
import matplotlib.pyplot as plt

def plot_stacked_bars(hnet_erors, honet_errors, fontsize=36, width=0.35):
    below = honet_errors
    above = below - hnet_erors

    N = 5
    ind = np.arange(N)  # the x locations for the groups
      # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, below, width, color='C1')
    p2 = plt.bar(ind, above, width, bottom=below)

    plt.ylabel('Scores')
    plt.title('Scores by group and gender')
    plt.xticks(ind, ('Crackers', 'Mustard', 'Orange', 'Woodblock', 'Total'))
    plt.yticks(np.arange(0, 40, 5))
    plt.legend((p1[0], p2[0]), ('HONet', 'HNet'), fontsize=fontsize)

    plt.show()

hnet_errors = np.array([36, 27, 25, 32, 30])
honet_errors = np.array([31, 24, 25, 29, 28])
plot_stacked_bars(hnet_errors, honet_errors)

hnet_rgb_errors = np.array([36, 27, 25, 32, 28])
honet_rgb_errors = np.array([33, 27, 27, 29, 27])
plot_stacked_bars(hnet_errors, honet_errors)