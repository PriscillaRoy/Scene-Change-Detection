import numpy as np
import cv2
from scipy import stats


import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.patches as patches


class FittingModels:

    def compute_cov(self,x,y):
        matrix = np.column_stack([x, y])
        matrix = matrix.astype(float)
        matrix -= matrix.mean(axis=0)
        cov = np.dot(matrix.T, matrix.conj()) / (len(x) - 1)
        return  cov

    def compute_mean(self, x):
        sum = 0
        for i in range(0, len(x)):
            sum = sum + x[i]
        mean = sum / len(x)
        return (mean)

    def fig_conv(self,fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data

    def ThreeD(self, a, b, c):
        lst = [[['#' for col in range(a)] for col in range(b)] for row in range(c)]
        return lst

    def create_dataPoints(self, image1, image2):
        #Converting the images to grayscale
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        max_row = len(gray_image1)
        max_col = len(gray_image1[0])
        data_points = self.ThreeD(2, max_col, max_row)

        for row in range(0, max_row):
            for col in range(0, max_col):
                data_points[row][col] = [gray_image1[row, col], gray_image2[row, col]]
        #A 3D matrix of datapoints of both the images
        return data_points

    def plot_data(self, data_points):
        X = []
        Y = []
        # Getting the parameters for the plot from the data points

        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                X.append(data_points[row][col][0])
                Y.append(data_points[row][col][1])

        # Plotting the data
        fig_1 = plt.figure()
        plot = fig_1.add_subplot(111)
        plot.scatter(X, Y)
        plt.xlabel('Image 1 Intensities')
        plt.ylabel('Image 2 Intensities')
        plt.title('Plotted Data')
        output_image_plot = self.fig_conv(fig_1)

        # test_image_return = np.zeros((100, 100), np.uint8) #test remove before submission
        return output_image_plot

    def fit_line_ls(self, data_points, threshold):

        image1_intensity = []  # Image1 intensities
        image2_intensity = []  # Image2 intensities
        data_points_original = []  # Image1 & Image2 intensities
        #Converting intensities for plot
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                image1_intensity.append(data_points[row][col][0])
                image2_intensity.append(data_points[row][col][1])
                data_points_original.append([data_points[row][col][0], data_points[row][col][1]])
        # Least Squares
        image1_mean = self.compute_mean(image1_intensity)
        image2_mean = self.compute_mean(image2_intensity)

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

        x = np.linspace(min_intensity, max_intensity, 1000)
        y = c_intercept + m_slope * x

        #Plot for LS with Regression Line
        fig_2 = plt.figure()
        plot = fig_2.add_subplot(111)
        plot.plot(x, y, color='#58b970', label='Regression Line')
        plt.scatter(image1_intensity, image2_intensity, label='Data Points')

        plt.xlabel('Image 1 Intensities')
        plt.ylabel('Image 2 Intensities')
        plt.legend()
        plt.title('Least Squares Line Fitting')
        line_fitting_image = self.fig_conv(fig_2)

        image1_thresholded = list()
        image2_thresholded = list()
        thresholded_intensities = list()

        #Finding the outliers
        new_segmented_image = [-1] * len(data_points_original)
        for i in range(0, len(image1_intensity)):
            perp_dist = abs((m_slope * image1_intensity[i]) - (image2_intensity[i]) + c_intercept) / np.sqrt(
                pow(m_slope, 2) + 1)
            if (perp_dist > threshold):
                image1_thresholded.append(image1_intensity[i])
                image2_thresholded.append(image2_intensity[i])
                thresholded_intensities.append([image1_intensity[i], image2_intensity[i]])
                new_segmented_image[i] = 255
            else:
                new_segmented_image[i] = 0

        # Plotting thresholded data points
        fig_3 = plt.figure()
        plot = fig_3.add_subplot(111)
        plt.scatter(image1_thresholded, image2_thresholded, label='Outliers')
        plt.xlabel('Image 1 Intensities')
        plt.ylabel('Image 2 Intensities')
        plt.legend()
        plt.title('Least Squares Thresholded Points')
        ls_thresholded_image = self.fig_conv(fig_3)

        # Reshaping the output image
        output_image = np.reshape(new_segmented_image, (len(data_points), len(data_points[0])))

        # Erosion and Dilation
        kernel = np.ones((2, 2), np.uint8)
        eroded_image = cv2.erode((output_image * 1.0).astype(np.float32), kernel, iterations=1)
        # test_image_return = np.zeros((100,100), np.uint8)#test remove before submission
        print("LS")
        return (line_fitting_image, ls_thresholded_image, eroded_image)

    def fit_line_robust(self, data_points, threshold):
        image1_intensity = []  # Image1 intensities
        image2_intensity = []  # Image2 intensities
        data_points_original = []  # Image1 & Image2 intensities
        # Converting intensities for plot
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                image1_intensity.append(data_points[row][col][0])
                image2_intensity.append(data_points[row][col][1])
                data_points_original.append([data_points[row][col][0], data_points[row][col][1]])

        #Robust Line Fitting
        points_matrix = np.asarray(data_points_original)
        vx, vy, x, y = cv2.fitLine(points_matrix, cv2.DIST_L2, 0, 0.01, 0.01)
        m_slope = (vy / vx)
        c_intercept = y - (m_slope * x) # Line Equation y = mx + c -> c = y - mx

        max_intensity = np.max(image1_intensity)
        min_intensity = np.min(image1_intensity)

        x_axis = np.linspace(min_intensity, max_intensity, 1000)
        y_axis = np.array(c_intercept + m_slope * x_axis)

        #Plotting the Regression Line for Robust Fitting
        fig_2 = plt.figure()
        plot = fig_2.add_subplot(111)
        plot.plot(x_axis, y_axis, color='#58b970', label='Regression Line')
        plt.scatter(image1_intensity, image2_intensity, label='Data Points')

        plt.xlabel('Image 1 Intensities')
        plt.ylabel('Image 2 Intensities')
        plt.legend()
        plt.title('Robust Line Fitting')
        robust_fitting_image = self.fig_conv(fig_2)

        image1_thresholded = list()
        image2_thresholded = list()
        thresholded_intensities = list()

        #Finding the outliers
        new_segmented_image = [-1] * len(data_points_original)
        for i in range(0, len(image1_intensity)):
            perp_dist = abs((m_slope * image1_intensity[i]) - (image2_intensity[i]) + c_intercept) / np.sqrt(
                pow(m_slope, 2) + 1)
            if (perp_dist > threshold):
                image1_thresholded.append(image1_intensity[i])
                image2_thresholded.append(image2_intensity[i])
                thresholded_intensities.append([image1_intensity[i], image2_intensity[i]])
                new_segmented_image[i] = 255
            else:
                new_segmented_image[i] = 0

        # Plotting thresholded data points
        fig_3 = plt.figure()
        plot = fig_3.add_subplot(111)
        plot.scatter(image1_thresholded, image2_thresholded, label='Outliers')
        plt.xlabel('Image 1 Intensities')
        plt.ylabel('Image 2 Intensities')
        plt.legend()
        plt.title('Robust Fitting Thresholded Points')
        robust_thresholded_image = self.fig_conv(fig_3)

        # Reshaping the output image
        output_image = np.reshape(new_segmented_image, (len(data_points), len(data_points[0])))

        # Erosion and Dilation
        kernel = np.ones((2, 2), np.uint8)
        eroded_image = cv2.erode((output_image * 1.0).astype(np.float32), kernel, iterations=1)

        #test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("RO")
        return (robust_fitting_image, robust_thresholded_image, eroded_image)

    def fit_gaussian(self, data_points, threshold):

        image1_intensity = []  # Image1 intensities
        image2_intensity = []  # Image2 intensities
        data_points_original = []  # Image1 & Image2 intensities
        # Converting intensities for plot
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                image1_intensity.append(data_points[row][col][0])
                image2_intensity.append(data_points[row][col][1])
                data_points_original.append([data_points[row][col][0], data_points[row][col][1]])
        # Gaussian Line Fitting
        image1_mean = self.compute_mean(image1_intensity)  # Mean in x direction
        print(image1_mean)
        image2_mean = self.compute_mean(image2_intensity)  # Mean in y direction
        print(image2_mean)
        covariance = self.compute_cov(image1_intensity, image2_intensity)# Covariance

        # Plotting the Fitting Line for Gaussian Fitting
        fig_2 = plt.figure()
        ax2 = fig_2.add_subplot(111, aspect='equal')
        ax2.scatter(image1_intensity, image2_intensity, label = 'Data Points')
        v, w = np.linalg.eigh(covariance) # Finding eigen values, eigen vectors for construction of Ellipse
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi
        v = 2 * np.sqrt(v * stats.chi2.ppf(threshold, 2))  # get size of ellipse according to threshold
        ax2.add_patch( patches.Ellipse(xy=(image1_mean, image2_mean), width = v[0],height= v[1], angle = 180 + angle , facecolor='none',label = 'classifier line',
                                       fill=False, ls='dashed', lw=1.5)
                       )

        plt.xlabel('Image 1 Intensities')
        plt.ylabel('Image 2 Intensities')
        plt.legend()
        plt.title('Gaussian Fitting')
        gaussian_filter_image = self.fig_conv(fig_2)

        gauss_prob_list = list()

        #Finding the probability values of each point
        matrix_1 = np.linalg.inv(covariance)
        constant_val = 1 / (2 * (22 / 7)*np.sqrt(np.linalg.det(covariance)))
        new_segmented_image = [-1] * len(data_points_original)
        for i in range(0, len(image1_intensity)):
            matrix_2 = np.matrix([(image1_intensity[i] - image1_mean), (image2_intensity[i] - image2_mean)]) # [[xi - xmean],[yi - ymean]]
            matrix_3 = matrix_2.transpose()
            temp_matrix = matrix_2.dot(matrix_1).dot(matrix_3)
            gauss_prob = constant_val * np.exp((-1 / 2) * temp_matrix)
            gauss_prob_list.append(gauss_prob)

        image1_thresholded = list()
        image2_thresholded = list()
        new_gauss_prob_list = np.asarray(gauss_prob_list)
        #Normalizing the probabilities
        norm_gauss_prob_list = (new_gauss_prob_list - new_gauss_prob_list.min())/(new_gauss_prob_list.max()- new_gauss_prob_list.min())
        i = 0

        for gauss_prob in norm_gauss_prob_list:
            if ((gauss_prob) < abs(1-threshold)):
                image1_thresholded.append(image1_intensity[i])
                image2_thresholded.append(image2_intensity[i])
                new_segmented_image[i] = 255
            else:
                new_segmented_image[i] = 0
            i = i + 1

        # Plotting thresholded data points
        fig_3 = plt.figure()
        plot = fig_3.add_subplot(111)
        plot.scatter(image1_thresholded, image2_thresholded, label='Outliers')
        plt.xlabel('Image 1 Intensities')
        plt.ylabel('Image 2 Intensities')
        plt.legend()
        plt.title('Gaussian Fitting Thresholded Points')
        gaussian_thresholded_image = self.fig_conv(fig_3)

        # Reshaping the output image
        output_image = np.reshape(new_segmented_image, (len(data_points), len(data_points[0])))
        # Erosion and Dilation
        kernel = np.ones((4, 4), np.uint8)
        eroded_image = cv2.erode((output_image * 1.0).astype(np.float32), kernel, iterations=1)

        #test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("GA")
        return (gaussian_filter_image, gaussian_thresholded_image, eroded_image)