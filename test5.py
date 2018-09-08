import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
def fig2data( fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img( fig):
    buf = fig2data(fig)
    w, h, d = buf.shape
    return buf
    #return Image.frombytes("RGBA", (w, h), buf.tostring())

fig_2 = plt.figure()
plot = fig_2.add_subplot(111)
x = np.random.random((300,1))
y =  np.random.random((300,1))
plt.scatter(x,y)
buf = fig2img(fig_2)
print (buf)

