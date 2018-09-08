import numpy as np
import cv2
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

class FittingModels:


    def ThreeD(self,a, b, c):
        lst = [[['#' for col in range(a)] for col in range(b)] for row in range(c)]
        return lst


    def create_dataPoints(self, image1, image2):

        gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        max_row = len(gray_image1)
        max_col = len(gray_image1[0])
        data_points = self.ThreeD(2,max_col,max_row)
        i = 0

        for row in range(0,max_row):
            for col in range(0,max_col):
                data_points[row][col] = [gray_image1[row,col],gray_image2[row,col]]

        return data_points


    def plot_data(self, data_points):
        X = []
        Y = []
        #fig = plt.figure()
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                X.append(data_points[row][col][0])
                Y.append(data_points[row][col][1])
        print(len(X))
        #print(X)
        #print(Y)

        plt.scatter(X,Y)
        plt.show()

        #fig.canvas.draw()
        #w, h = fig.canvas.get_width_height()
        #buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
        #buf.shape = (w, h, 4)

        # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
        #buf = np.roll(buf, 3, axis=2)
        #return buf


        # Now we can save it to a numpy array.
        #data = np.fromstring(fig.canvas.tostring_rgb(blit=False), dtype=np.uint8, sep='')
        #data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

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
        X = []
        Y = []
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                X.append(data_points[row][col][0])
                Y.append(data_points[row][col][1])
        # Mean X and Y
        mean_x = np.mean(X)
        mean_y = np.mean(Y)

        # Total number of values
        m = len(X)

        # Using the formula to calculate b1 and b2
        numer = 0
        denom = 0
        for i in range(m):
            numer += (X[i] - mean_x) * (Y[i] - mean_y)
            denom += (X[i] - mean_x) ** 2
        b1 = numer / denom
        b0 = mean_y - (b1 * mean_x)

        # Print coefficients
        print(b1, b0)

        max_x = np.max(X)
        min_x = np.min(X)
        print(max_x,min_x)

        # Calculating line values x and y
        x = np.linspace(min_x, max_x, 1000)
        #print(x)
        y = b0 + b1 * x
        #print(y)

        # Ploting Line
        plt.plot(x, y, color='#58b970', label='Regression Line')
        # Ploting Scatter Points
        plt.scatter(X, Y, label='Scatter Plot')

        plt.xlabel('I1')
        plt.ylabel('I2')
        plt.legend()
        plt.show()

        data_points_x = list()
        data_points_y = list()
        data_points_thresholded = list()
        for i in range(0,len(X)):
            perp_dist = abs((b1*X[i]) - (Y[i]) + b0)/np.sqrt(pow(b1,2)+ 1)
            if (perp_dist > threshold):
                data_points_x.append(X[i])
                data_points_y.append(Y[i])
                #data_points_thresholded.append(X[i],Y[i])


        plt.scatter(data_points_x, data_points_y, label='Scatter Plot')
        plt.show()

        segmented_image = [[-1] * data_points[0] for _ in range(data_points)]
        n =0
        for row in range(0, len(data_points)):
            for col in range(0, len(data_points[0])):
                if(data_points_x[n] in ):

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
