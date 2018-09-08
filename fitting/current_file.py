import numpy as np
import cv2

import matplotlib
matplotlib.use('Agg')

from matplotlib import pyplot as plt
import matplotlib.patches as patches



class FittingModels:

    def compute_cov(self,x,y):
        N = len(x)
        X = np.column_stack([x, y])
        X = X.astype(float)
        X -= X.mean(axis=0)
        fact = N - 1
        cov = np.dot(X.T, X.conj()) / fact
        return  cov
    def compute_mean(self,x):
        sum = 0
        for i in range(0, len(x)):
            sum = sum + x[i]
        mean = sum/len(x)
        return (mean)

    def fig2data(self,fig):
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return data


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

        #Getting the parameters for the plot from the data points

        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                X.append(data_points[row][col][0])
                Y.append(data_points[row][col][1])

        #Plotting the data
        plt.scatter(X,Y)
        im = 'plotted_image.jpg'
        plt.savefig(im,dpi = 150)

        test_image_return = np.zeros((100, 100), np.uint8) #test remove before submission
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

        x = np.linspace(min_intensity, max_intensity, 1000)
        y = c_intercept + m_slope * x
        fig_2 = plt.figure()
        plot = fig_2.add_subplot(111)
        plot.plot(x, y, color='#58b970', label='Regression Line')
        plt.scatter(image1_intensity, image2_intensity, label='Scatter Plot')

        plt.xlabel('Intensity1')
        plt.ylabel('Intensity2')
        plt.legend()
        #ls_regression_line = self.fig2img(fig_2)
        #line_fitting_image = np.asarray(ls_regression_line)
        #fig_2 = plt.figure()
        #plot = fig_2.add_subplot(111)
        #x = np.random.random((300, 1))
        #y = np.random.random((300, 1))
        #plt.scatter(x, y)
        line_fitting_data = self.fig2buf(fig_2)
        line_fitting_image = np.asarray(line_fitting_data)

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
        #plt.scatter(image1_thresholded, image2_thresholded, label='Scatter Plot')
        #fig_3 = plt.figure()
        #plot = fig_3.add_subplot(111)
        #plot.scatter(image1_thresholded, image2_thresholded, label='Scatter Plot')
        #ls_thresholded_data = self.fig2img(fig_3)
        #ls_thresholded_image = np.asarray(ls_thresholded_data)

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
        kernel = np.ones((2, 2), np.uint8)
        eroded_image = cv2.erode((output_image*1.0).astype(np.float32), kernel, iterations=1)
        #eroded_image = erosion(np.asanyarray(output_image), selem=None, out=None, shift_x=False, shift_y=False)
        test_image_return = np.zeros((100,100), np.uint8)#test remove before submission
        print("LS")
        return (line_fitting_image, test_image_return, eroded_image)

    def fit_line_robust(self, data_points, threshold):
        image1_intensity = []  # Image1 intensities
        image2_intensity = []  # Image2 intensities
        data_points_original = []  # Image1 & Image2 intensities
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                image1_intensity.append(data_points[row][col][0])
                image2_intensity.append(data_points[row][col][1])
                data_points_original.append([data_points[row][col][0], data_points[row][col][1]])
        points_matrix = np.asarray(data_points_original)
        vx,vy,x,y = cv2.fitLine(points_matrix,cv2.DIST_L2,0,0.01,0.01)
        #Line Equation y = mx + c -> c = y - mx
        m_slope = (vy/vx)
        c_intercept = y -(m_slope*x)

        max_intensity = np.max(image1_intensity)
        min_intensity = np.min(image1_intensity)

        x_axis = np.linspace(min_intensity, max_intensity, 1000)
        y_axis = np.array(c_intercept + m_slope * x_axis)
        fig_2 = plt.figure()
        plot = fig_2.add_subplot(111)
        plot.plot(x_axis, y_axis, color='#58b970', label='Regression Line')
        plt.scatter(image1_intensity, image2_intensity, label='Scatter Plot')

        plt.xlabel('Intensity1')
        plt.ylabel('Intensity2')
        plt.legend()
        robust_regression_line = self.fig2img(fig_2)
        line_fitting_image = np.asarray(robust_regression_line)

        image1_thresholded = list()
        image2_thresholded = list()
        thresholded_intensities = list()
        for i in range(0, len(image1_intensity)):
            perp_dist = abs((m_slope * image1_intensity[i]) - (image2_intensity[i]) + c_intercept) / np.sqrt(
                pow(m_slope, 2) + 1)
            if (perp_dist > threshold):
                image1_thresholded.append(image1_intensity[i])
                image2_thresholded.append(image2_intensity[i])
                thresholded_intensities.append([image1_intensity[i], image2_intensity[i]])

        # Plotting thresholded data points
        plt.scatter(image1_thresholded, image2_thresholded, label='Scatter Plot')
        fig_3 = plt.figure()
        plot = fig_3.add_subplot(111)
        plot.scatter(image1_thresholded, image2_thresholded, label='Scatter Plot')
        robust_thresholded_data = self.fig2img(fig_3)
        robust_thresholded_image = np.asarray(robust_thresholded_data)

        # Making the segmented image
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
        # Reshaping the output image
        output_image = np.reshape(segmented_image, (len(data_points), len(data_points[0])))

        # Erosion and Dilation
        kernel = np.ones((2, 2), np.uint8)
        eroded_image = cv2.erode((output_image * 1.0).astype(np.float32), kernel, iterations=1)

        test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("RO")
        #return (line_fitting_image, robust_thresholded_image, eroded_image)
        return(test_image_return,test_image_return,test_image_return)
    def fit_gaussian(self, data_points, threshold):

        image1_intensity = [] # Image1 intensities
        image2_intensity = [] # Image2 intensities
        data_points_original =[] # Image1 & Image2 intensities
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                image1_intensity.append(data_points[row][col][0])
                image2_intensity.append(data_points[row][col][1])
                data_points_original.append([data_points[row][col][0],data_points[row][col][1]])
        # Gaussian
        # Mean
        image1_mean = self.compute_mean(image1_intensity) # Mean in x direction

        image2_mean = self.compute_mean(image2_intensity) # Mean in y direction

        # Covariance
        covariance = self.compute_cov(image1_intensity,image2_intensity)

        eigen1,eigen2 = np.linalg.eig(covariance)
        axis1 = 2 *3.2* np.sqrt(eigen1[0])
        axis2 = 2 *3.3* np.sqrt(eigen1[1])
        fig_2 = plt.figure()
        ax2 = fig_2.add_subplot(111, aspect='equal')
        ax2.scatter(image1_intensity,image2_intensity)
        ax2.add_patch(
            patches.Ellipse( xy =(image1_mean,image2_mean),width = axis1, height= axis2, angle = np.rad2deg(np.arccos(eigen2[0,0])),
                fill=False  # remove background
            )
        )
        gaussian_filter_data = self.fig2img(fig_2)
        gaussian_filter_image = np.asarray(gaussian_filter_data)


        image1_thresholded = list()
        image2_thresholded = list()
        thresholded_intensities = list()
        gauss_prob_list = list()
        matrix_1 =np.linalg.inv(covariance)

        for i in range(0, len(image1_intensity)):

            constant_val = np.sqrt(np.linalg.det(covariance))
            matrix_2 = np.matrix([(image1_intensity[i] - image1_mean),(image2_intensity[i] - image2_mean) ])
            matrix_3 = matrix_2.transpose()
            temp_matrix = matrix_2.dot(matrix_1).dot(matrix_3)
            gauss_prob =(1/(2*(22/7)*constant_val))* np.exp((-1/2)*temp_matrix)
            gauss_prob_list.append(gauss_prob)
            if (gauss_prob < threshold):
                image1_thresholded.append(image1_intensity[i])
                image2_thresholded.append(image2_intensity[i])
                thresholded_intensities.append([image1_intensity[i], image2_intensity[i]])
        fig_3 = plt.figure()
        plot = fig_3.add_subplot(111)
        plot.scatter(image1_thresholded, image2_thresholded, label='Scatter Plot')

        gaussian_thresholded_data = self.fig2img(fig_3)
        gaussian_thresholded_image = np.asarray(gaussian_thresholded_data)

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
        # Erosion and Dilation
        kernel = np.ones((2, 2), np.uint8)
        eroded_image = cv2.erode((output_image * 1.0).astype(np.float32), kernel, iterations=1)

        test_image_return = np.zeros((100, 100), np.uint8)  # test remove before submission
        print("GA")
        #return (img, gaussian_thresholded_image, eroded_image)
        return(test_image_return,test_image_return,test_image_return)