#import matplotlib.pyplot
import matplotlib
matplotlib.use('TkAgg')
import numpy
from matplotlib import pyplot as plt
from PIL import Image
import cv2


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = numpy.fromstring(fig.canvas.tostring_argb(), dtype=numpy.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll(buf, 3, axis=2)
    return buf



def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    print(Image.frombytes("RGBA", (w, h), buf.tostring()))
    return Image.frombytes("RGBA", (w, h), buf.tostring())

# Generate a figure with matplotlib</font>
figure = plt.figure()
plot = figure.add_subplot(111)

# draw a cardinal sine plot
x = numpy.arange(1, 100, 0.1)
y = numpy.sin(x) / x
plot.plot(x, y)
#matplotlib.pyplot.show()
im = fig2img ( figure )
output_dir = 'test_output/'
output_path = output_dir + "data_test" + ".jpg"
op= numpy.asarray(im)
cv2.imwrite(output_path,op)
im.show()