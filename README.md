The main goal of this project is given a leaf image of tomato, pepper or potato, the output is the kind of disease (or healthy if not).
The dataset contains almost 20,000 images. Each image is labeled with the kind of it's disease. 
In this project, I classified the data with multiclass classification and CNN. Multiclass classification is  classifying instances into one of three or more classes, so it's a perfect fit for our needs.
Data are splitted into ~80% training, and ~20% test.
The prediction model is saved into a file called "plantDieasesPrediction", and we do the test on it.
Finally, the input image (the image of the leaf we wanted to test) is being predicted from this model.


How to run:
1)Download the zip folder of the dataset I sent by Email.
2)Unzip the folder and copy the path of the folder.
3)Follow the instruction on the code file (the .py file) and paste the path when it's needed (there are comments where you need to paste it. line: 28, 31, 34, 41, 51).
4)Copy the path of the image you want to test and paste it to line 143.
5)Run the code.


----------------------------------------------------------------------------------------------------------------------------------------------------

For the data split, I used "train_test_split(X, Y, test_size=0.2, random_state=0,shuffle=True)". As shown in parameter "test_size = 0.2", it says the 20% of the data will be splitted to the test, and the rest is for train.
For speculation parameters, I used a class called "myCallBack". This class contains a function that runs when training the model. It stops the training stops when the model reaches 90% accuracy or more (as I used 90% accuracy as a high accuracy). The reasons why 90% is a high accuracy is first of all the dataset contains a small ammount of images (20,000 is not a high ammount). Secondly, the most of the data are similar to each other, so it makes it harder to get accurate. Finally, there are more images in some labels than the others, as they are not equall.  
For the prediction of the model and testing the data, I used "model.predict(X_test, batch_size = 32)". The parameter "X_test" is given to do the prediction on the test data.
Finally, for the input image to predict for which disease it belongs, I used "model.predict(img_array)". This function predicts the result based on the model (called "model") I trained before and saved.
