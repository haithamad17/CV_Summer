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
