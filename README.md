Customer churn analysis with data from a telecom company
Project made 8-10-25

My goal with this project was to use everything I have learned in my previous projects and apply it to make a complete project. For each of my previous projects, I would find a dataset that fit my learning goal for that project (ex: encoding categories, model selection, data visualizations). This time, I am using what I have learned to analyze a dataset.

Dataset used: "Iranian Churn." UCI Machine Learning Repository, 2020, https://doi.org/10.24432/C5JW3Z.

Libraries used: pandas, numpy, matplotlib, seaborn, keras, sklearn

Dataset columns: Call Failure	Complains	Subscription Length	Charge Amount	Seconds of Use	Frequency of use	Frequency of SMS	Distinct Called Numbers	Age Group	Tariff Plan	Status	Age	Customer Value	Churn (label)

EDA and Preprocessing:

First, I created a correlation matrix of all the columns in the dataset. This showed me how well the features were correlated to the label. I wasn't expecting high correlation values because the label is binary (non-continuous). However, I found that "complains" and "status" have good correlation with churn, which makes sense because customers that complain more on calls are less likely to stay with the company, and customers that are inactive are also less likely to stay. Because the features did not have strong correlation with the label, I think a simple logistic regression would not be accurate enough for this model. To avoid multicollinearity, I looked for features that had very stong correlation with each other. Two features in the dataset, "age" and "age group", had a correlation of 0.96 with each other. I decided to drop the "Age" oclumn because that column was never specified in the dataset description. "Frequency of use" and "seconds of use" had a correlation of 0.94, meaning I would need to drop one of the columns. However, I wanted to transform those features into something more meaningful, so I created a new column called "avg_call_duration" which divided the seconds by the frequency to get an average of how long each customer spent per phone call. I dropped the "frequency of use" column, but I kept the "seconds of use" column because I think knowing the total time a customer spends on phone calls is still important in customer churn.

Next, I looked at the label distribution and found that there were 2655 instances of the negative class (non-churn) and 495 instances of the positive class (churn). To allow the model to learn a more even amount from both classes, I downsampled the negtive class to be double the size of the positive class.

After that, I looked at the distributions of each of the features to see how to scale them. What was interesting about this dataset is that most features followed the power law distribution, and only a couple followed a normal distribution. In all of my previous projects, I would first normalize all of the features and then split the data into training and testing, but now I figured out that this is a common form of data leakage. To keep the test data/stats completely separate from the training data, I first split the data into 80% train 20% test and then scaled the features. To scale them, I first fit the scaler objects to only the train set, and then I transformed the train and test sets using those objects. One issue I found when doing this was when I was specifying which columns to scale, I kept getting errors because it couldn't find that column name. After a while of debugging, I found that whoever made the dataset put double spaces in some feature names and single spaces in others.


Model creation:

I chose to make a neural net to train on the data rather than a logistic regression because of low correlation values between features and the label. The hidden layers had 64, 32, and 8 nodes respectively, all with ReLU activation. The final output had a sigmoid transform to output a probability. The model trained through 15 epochs with a batch size of 64. I tested how changing the loss function affected the performance of the network, and I found that training on mse loss was slightly better than cross entropy loss.

Model evaluation:

The test loss (mse) was 0.057, and the confusion matrix of the test set is as follows: TP: 90 TN: 185 FP: 12 FN: 10 
The training loss was 0.0445, meaning the model overfit very slightly.
The model had an accuracy of 0.93, precision of 0.95, and recall of 0.94. It is important that the model predicts false positives more than false negatives in this context because it is worse to lose a customer than to spend extra trying to keep a customer that was nevery going to leave.
Finally, I made an ROC curve and found the AUC to be 0.975, which means the model was making confident predictions.





