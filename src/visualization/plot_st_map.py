import numpy as np
import pandas as pd
import heartpy as hp

import matplotlib.pyplot as plt

st_3 = np.load('test.npy')

fig = plt.figure(figsize=(5, 10))
plt.pcolormesh(st_3[:, :, 0], cmap='cool')
plt.xlabel('ROI')
plt.xticks([0.5, 1.5, 2.5], ['whole_body', 'face', 'chest'])
plt.ylabel('Frames')
plt.savefig('test.pdf')
plt.show()
