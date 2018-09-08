import numpy as np
import cv2

import skimage
import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
from PIL import Image
from skimage.morphology import erosion

class FittingModels:
    def fig2data(self,fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        return buf

    def fig2img(self,fig):

        buf = self.fig2data(fig)
        w, h, d = buf.shape
        return Image.frombytes("RGBA", (w, h), buf.tostring())

    def ThreeD(self,a, b, c):
        lst = [[['#' for col in range(a)] for col in range(b)] for row in range(c)]
        return lst


    def create_dataPoints(self, image1, image2):

        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        max_row = len(gray_image1)
        max_col = len(gray_image1[0])
        data_points = self.ThreeD(2,max_col,max_row)

        for row in range(0,max_row):
            for col in range(0,max_col):
                data_points[row][col] = [gray_image1[row,col],gray_image2[row,col]]

        return data_points


    def plot_data(self, data_points):
        X = []
        Y = []

        fig_1 = plt.figure()
        plot = fig_1.add_subplot(111)

        #Getting the parameters for the plot from the data points

        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                X.append(data_points[row][col][0])
                Y.append(data_points[row][col][1])
        print(len(X))
        #Plotting the data
        plot.scatter(X,Y)
        im = self.fig2img(fig_1)
        output_image_plot = np.asarray(im)

        plt.show()

        #test_image_return = np.zeros((100, 100), np.uint8) #test remove before submission
        return output_image_plot

    def fit_line_ls(self, data_points, threshold):

        image1_intensity = [] # Image1 intensities
        image2_intensity = [] # Image2 intensities
        data_points_original =[] # Image1 & Image2 intensities
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                image1_intensity.append(data_points[row][col][0])
                image2_intensity.append(data_points[row][col][1])
                data_points_original.append([data_points[row][col][0],data_points[row][col][1]])
        # Least Squares
        image1_mean = np.mean(image1_intensity)
        image2_mean = np.mean(image2_intensity)

        number_of_intensities = len(image1_intensity)
        n = 0
        d = 0
        for i in range(number_of_intensities):
            n += (image1_intensity[i] - image1_mean) * (image2_intensity[i] - image2_mean)
            d += (image1_intensity[i] - image1_mean) ** 2
        m_slope = n / d
        c_intercept = image2_mean - (m_slope * image1_mean)

        max_intensity = np.max(image1_intensity)
        min_intensity = np.min(image1_intensity)
        print(max_intensity,min_intensity)

        x = np.linspace(min_intensity, max_intensity, 1000)
        y = c_intercept + m_slope * x
        fig_2 = plt.figure()
        plot = fig_2.add_subplot(111)
        plot.plot(x, y, color='#58b970', label='Regression Line')
        plt.scatter(image1_intensity, image2_intensity, label='Scatter Plot')

        plt.xlabel('Intensity1')
        plt.ylabel('Intensity2')
        plt.legend()
        ls_regression_line = self.fig2img(fig_2)
        line_fitting_image = np.asarray(ls_regression_line)

        #plt.show()


        image1_thresholded = list()
        image2_thresholded = list()
        thresholded_intensities = list()
        for i in range(0,len(image1_intensity)):
            perp_dist = abs((m_slope*image1_intensity[i]) - (image2_intensity[i]) + c_intercept)/np.sqrt(pow(m_slope,2)+ 1)
            if (perp_dist > threshold):
                image1_thresholded.append(image1_intensity[i])
                image2_thresholded.append(image2_intensity[i])
                thresholded_intensities.append([image1_intensity[i],image2_intensity[i]])

        #Plotting thresholded data points
        plt.scatter(image1_thresholded, image2_thresholded, label='Scatter Plot')
        plt.show()
        fig_3 = plt.figure()
        plot = fig_3.add_subplot(111)
        plot.scatter(image1_thresholded, image2_thresholded, label='Scatter Plot')
        ls_thresholded_data = self.fig2img(fig_3)
        ls_thresholded_image = np.asarray(ls_thresholded_data)

        #Making the segmented image
        for i in range(0, len(thresholded_intensities)):
            if (thresholded_intensities[i] in data_points_original):
                index = data_points_original.index(thresholded_intensities[i])
                data_points_original[index] = [255, 255]

        segmented_image = []
        for i in range(0, len(data_points_original)):
            if (data_points_original[i] == [255, 255]):
                segmented_image.append(255)
                continue
            else:
                data_points_original[i] = [0, 0]
                segmented_image.append(0)
        #Reshaping the output image
        output_image= np.reshape(segmented_image,(len(data_points),len(data_points[0])))

        #Erosion and Dilation
        eroded_image = erosion(np.asanyarray(output_image), selem=None, out=None, shift_x=False, shift_y=False)
        #test_image_return = np.zeros((100,100), np.uint8)#test remove before submission
        print("LS")
        return (line_fitting_image, ls_thresholded_image, eroded_image)

    def fit_line_robust(self, data_points, threshold):
        """ Fits a line to the given data points using robust estimators
        :param data_points: a list of data points
        :param threshold: a threshold value (if > threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * A segmented image
        """
        test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("RO")
        return (test_image_return, test_image_return, test_image_return)

    def fit_gaussian(self, data_points, threshold):
        """ Fits the data points to a gaussian
        :param data_points: a list of data points
        :param threshold: a threshold value (if < threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the gaussian along with the data points
                    * The thresholded image
                    * A segmented image
        """
        test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("GA")
        return (test_image_return, test_image_return, test_image_return)
