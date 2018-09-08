import numpy as np
import cv2
import matplotlib.pyplot as plt

class FittingModels:

    def create_dataPoints(self, image1, image2):
        """ Creates a list of data points given two images
        :param image1: first image
        :param image2: second image
        :return: a list of data points
        """
        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        max_row = len(gray_image1)
        max_col = len(gray_image1[0])
        data_points = [[-1] * (max_col*max_row) for _ in range(2)]
        i = 0

        for row in range(0,max_row):
            for col in range(0,max_col):
                data_points[0][i] = gray_image1[row,col]
                data_points[1][i] = gray_image2[row,col]
                i = i + 1

        #print(data_points)
        return data_points


    def plot_data(self, data_points):
        """ Plots the data points
        :param data_points:
        :return: an image
        """
        plt.scatter(data_points[0],data_points[1],zorder = 2)
        #plt.show()
        test_image_return = np.zeros((100, 100), np.uint8) #test remove before submission
        return test_image_return

    def fit_line_ls(self, data_points, threshold):
        """ Fits a line to the given data points using least squares
        :param data_points: a list of data points
        :param threshold: a threshold value (if > threshold, imples outlier)
        :return: a tuple containing the followings:
                    * An image showing the line along with the data points
                    * The thresholded image
                    * A segmented image
        """

        # Mean X and Y
        mean_x = np.mean(data_points[0])
        mean_y = np.mean(data_points[1])

        # Total number of values
        m = len(data_points[0])

        # Using the formula to calculate b1 and b2
        numer = 0
        denom = 0
        for i in range(m):
            numer += (data_points[0][i] - mean_x) * (data_points[1][i] - mean_y)
            denom += (data_points[0][i] - mean_x) ** 2
        b1 = numer / denom
        b0 = mean_y - (b1 * mean_x)

        # Print coefficients
        print(b1, b0)

        max_x = np.max(data_points[0])
        min_x = np.min(data_points[0])

        # Calculating line values x and y
        x = np.linspace(min_x, max_x, 1000)
        y = b0 + b1 * x

        # Ploting Line
        plt.plot(x, y, color='#58b970', label='Regression Line')
        # Ploting Scatter Points
        plt.scatter(data_points[0], data_points[1], label='Scatter Plot')

        plt.xlabel('I1')
        plt.ylabel('I2')
        plt.legend()
        plt.show()

        data_points_x = list()
        data_points_y = list()
        for i in range(0,len(data_points[0])):
            perp_dist = abs((b1*data_points[0][i]) - (data_points[1][i]) + b0)/sqrt(pow(b1)+ 1)
            if (perp_dist > threshold):
                data_points_x.append(data_points[0][i])
                data_points_y.append(data_points_y[1][i])

        segmented_image =



        test_image_return = np.zeros((100,100), np.uint8)#test remove before submission
        print("LS")
        return (test_image_return, test_image_return, test_image_return)

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
