# Computer Vision
Scene Change Detection
See Homework - 2 description file (Homework-2.pdf) for details.

Create data points:
(1 Pts.) Write code to create the data points

  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "create_dataPoints". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns an object with data points (List, tuple,...)
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hw2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m TL -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m TL -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
-------------
Plot data points:
(1 Pts.) Write code to plot the data points on an image

  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "plot_data". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns an image with datapoints plotted
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m TL -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m TL -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
-------------
Total least square fitting:
(4 Pts.) Write code to fit a line using total least square, identify outliers and segmented image with the changes pixels.
  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "fit_line_tls". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns a tuple of three images (fitted line, thresholded image and segmented image) 
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m TL -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m TL -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
 -------------
 Robust estimation:
(3 Pts.) Write code to fit a line using robust estimators, identify outliers and segmented image with the changes pixels.
  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "fit_line_robust". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns a tuple of three images (fitted line, thresholded image and segmented image) 
  - For this part of the assignment, please implement your own code for all computations.
  - Exception, You are welcome to use the fitLinea function from opencv
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m RO -t 20
  
            python cv_hm2.py -i1 image1 -i2 image2 -m RO -t 20
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
  -------------
 Gaussian Fitting:
(4 Pts.) Write code to fit a gaussian model, identify outliers and segmented image with the changes pixels.
  - Starter code available in directory fitting/
  - fitting/models.py: Put your code in "fit_gaussian". You are welcome to add more function or add nested functions with in the function.
  - Make sure the function returns a tuple of three images (fitted line, thresholded image and segmented image) 
  - For this part of the assignment, please implement your own code for all computations, do not use inbuilt functions like from numpy, opencv or other libraries for clustering or segmentation.
  - Describe your method and findings in your report
  - This part of the assignment can be run using cv_hm2.py (there is no need to edit this file)
  - Usage: 
  
            ./cv_hm2 -i1 image1 -i2 image2 -m GA -t 0.7
  
            python cv_hm2.py -i1 image1 -i2 image2 -m GA -t 0.7
  - Please make sure your code runs when you run the above command from prompt/terminal
  - Any output images or files must be saved to "output/" folder (cv_hm2.py automatically does this)
  
  -------------
3. (2 Pts.) Describe your method and report you findings in your report for each of the fittings in assignemnt.
  - Your report should accompany your code. 
  - Include a word/pdf file in the repository.
  - In your report also decribe the following points
  
    a. What are the parameters that influence your algorithm? Explain their effect?
    
    b. Does one of the fitting models work better than the othet. Explain?
    
    c. What is the objective of this implementations? Which model works best and what are their parameters? Explain?
  
  -------------

Three sets of images are provided for testing
OpenCV: You can use opencv functions to load, display images.

Make sure your final submission is running on circleci. 
The TA will use CircleCI output and your github code for grading. 
TA will not be able to grade if the code does not run on circle CI.

Common reasons for failure.

Do not use any 3rd party libraries or functions.
Do not display images in your final submission. 
Example, cv2.imshow(), cv2.waitkey(), cv2.NamedWindow will make the circle ci fail.

PS. Files not to be changed: requirements.txt and .circleci directory 
If you do not like the structure, you are welcome to change the over all code, under two stipulations:

1. the homework has to run using command

  python cv_hm2.py -i1 image1 i2 image 2 -m TL -t 20,
  python cv_hm2.py -i1 image1 i2 image 2 -m RO it 20
  and
  python cv_hm2.py -i1 image1 i2 image 2 -m GA -t 20
 

  
2. Any output file or image should be written to output/ folder

The TA will only be able to see your results if these two conditions are met

1. Data Points         - 1 Pts.
2. Plotting            - 1 Pts.
3. Total Least Squares - 4 Pts
4. Robust Fitting      - 3 Pts
5. Gaussian Fitting    - 4 Pts
6. Report              - 2 Pts

    Total              - 15 Pts.

----------------------
