**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_1.jpg "Center Image"
[image2]: ./examples/left_1.jpg "Left Image"
[image3]: ./examples/right_1.jpg "Right Image"
[image4]: ./examples/trouble_red.jpg "Red markers Image"
[image5]: ./examples/bridge.jpg "Bridge Iamge"
[image6]: ./examples/track2_1.jpg "Track 2 Image"
[image7]: ./examples/non_flip.jpg "Non-Flipped Image"
[image8]: ./examples/flip.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode.

My project includes the following files:
* bin/drive/py (drive.py): for driving the car in autonomous mode
* bin/train_model.py (model.py): containing the script to create and train the model
* final_model.h5 (model.h5): containing a trained convolution neural network
* writeup_report.md: summarizing the results
* bc_helper: a module of helper code which I used while iterating/testing during the project. 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python ./bin/drive.py final_model.h5
```

####3. Submission code is usable and readable

The train_model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. Additional code for loading the specific data is in the bc_helper module. In this module I load different datasets, save the data to s3 and construct a generator. 

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is very similar to the model Nvidia used in their [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). The model consists of:
	- Crop
	- Normalization
	- CNN(s=2x2, k=5x5, d=24, relu=True)
	- CNN(s=2x2, k=5x5, d=36, relu=True)
	- CNN(s=2x2, k=5x5, d=48, relu=True)
	- CNN(s=1x1, k=3x3, d=64, relu=True)
	- CNN(s=1x1, k=3x3, d=64, relu=True)
	- Flatten
	- FCC(d=100, relu=True, dropout = 0.5)
	- FCC(d=50, relu=True, dropout = 0.5)
	- FCC(d=10, relu=True, dropout = 0.5)
	- FCC(d=1)

####2. Attempts to reduce overfitting in the model

The model uses dropout after each fully connected layer to reduce overfitting. I also doubled thenumber of images I had by flipping every image horizontally. This should reove any bias towards turning in a certain direction.

The model was trained and validated on different data sets to ensure the model was not overfit. I also included two laps from the second course to help generalize the model. Overall I found accuracy to to a poor reflection of a successful model in this project. For example, if you are on the edge of the road and need to recover, a small change (-0.1 vs. 0.1) could result in wildly different results on the track. 

The best test was running the model through the simulator on the first track and ensuring it stayed on the track/did not crash.

Unfortunetly, at this time I have not collected enough data on the second track and my car has a hard time driving there. 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train_model.py line 81).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the starter data provided by Udacity as a starting point and then recorded more data for trouble areas the car struggled with and some recovery data. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

I felt like that hardest part of this project was actually the data collection and selection. I started out by using the Udacity starter data and used Nvidia's trick of using the left and right images for recovery data. I used a sterring angle delta of 0.4. This hyper parameter could be played with more. 

I employed a model similar to Nvidia's model as I felt there model was probably properly sized based on thier success driving a car. My model was different from their model in a few ways. I first cropped the images to remove the sky and the tip of the car. This helped make the left and right camera images look more like the center camera. I then scaled the pixel values to be centered at 0 and between -0.5 and 0.5. Nvidia also normalized their model but they did not explain how. The next 8 layers (CCNx5 FCx3) were all the same. Nvidia did not discuss if or how they introduced non-liniarities but I decided to use a relu after each step of the model. I also added dropout in between each fully connected layer to reduce overfitting.

I split my data into 70% training and 30% validation and compared the mean squared errors. They were similar for the most part when using 0.5 dropout between each fully connected layer. 

The final test was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specifically when the road switched from yellow lane lines to red markers. To improve the driving behavior in these cases, I collected more data for these specific "trouble areas". My data was pretty heavily skewed towards steering angles of 0 before adding this data. Because of this, adding data for the "trouble areas" made sense because it increased the percentage of data collected in edge cases and on turns. After adding extra data for these cases the model preformed much better. For some extra generalization I also collected some driving data from track two. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

	- Crop
	- Normalization
	- CNN(s=2x2, k=5x5, d=24, relu=True)
	- CNN(s=2x2, k=5x5, d=36, relu=True)
	- CNN(s=2x2, k=5x5, d=48, relu=True)
	- CNN(s=1x1, k=3x3, d=64, relu=True)
	- CNN(s=1x1, k=3x3, d=64, relu=True)
	- Flatten
	- FCC(d=100, relu=True, dropout = 0.5)
	- FCC(d=50, relu=True, dropout = 0.5)
	- FCC(d=10, relu=True, dropout = 0.5)
	- FCC(d=1)

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the center lane driving of the starter data. Here is an example image of center lane driving:

![alt text][image1]

I then used the left and right images from the starter data as recovery images from the left side and right sides of the road back to center so that the vehicle would learn to move back to the center if it drifted. These images show what a recovery looks like starting from left nad right sides respectively:

![alt text][image2]
![alt text][image3]

After testing the case with just this data, I saw that I struggled on the red marked turns and crossing the bridge. To help fix this I specifically recorded images from these areas:

![alt text][image4]
![alt text][image5]

I then recorded data from the second track to help my model generalize to reccognizing road edges and now to navigate them. 

![alt text][image6]

To augment the data sat, I also flipped images and angles thinking that this would remove any bias the model had for guessing postitive or negative sterring angles. For example, here is an image not flipped and flipped:

![alt text][image7]
![alt text][image8]

After the collection process, I had 63178 data points. The preprocessing I did was minimal and took place as part of the Keras model. I normalized the images to be centered at 0 and between -0.5 and 0.5 and I cropped the images to hide the sky and the top of the car.

I finally randomly shuffled the data set and put 30% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model on 30 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.

