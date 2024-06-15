import numpy as np
import matplotlib.pyplot as plt

phi = 2.0 * np.pi * np.random.rand(512)

cos_theta = 2.0 * np.random.rand(512) - 1.0
sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 0.0))

x = sin_theta * np.cos(phi)
y = sin_theta * np.sin(phi)
z = cos_theta


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(x, y, z)

plt.show()