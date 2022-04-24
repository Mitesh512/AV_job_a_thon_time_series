# AV_job_a_thon_time_series

Problem Statement
ABC is a car rental company based out of Bangalore. It rents cars for both in and out stations at affordable prices. The users can rent different types of cars like Sedans, Hatchbacks, SUVs and MUVs, Minivans and so on.

In recent times, the demand for cars is on the rise. As a result, the company would like to tackle the problem of supply and demand. The ultimate goal of the company is to strike the balance between the supply and demand inorder to meet the user expectations. 

The company has collected the details of each rental. Based on the past data, the company would like to forecast the demand of car rentals on an hourly basis. 


Objective
The main objective of the problem is to develop the machine learning approach to forecast the demand of car rentals on an hourly basis.


Data Dictionary
You are provided with 3 files - train.csv, test.csv and sample_submission.csv

Training set

train.csv contains the hourly demand of car rentals from August 2018 to February 2021.


Variable

Description

date

Date (yyyy-mm-dd)

hour

Hour of the day

demand

No. of car rentals in a hour


Test set
test.csv contains only 2 variables: date and hour. You need to predict the hourly demand of car rentals for the next 1 year i.e. from March 2021 to March 2022.


Variable

Description

date

Date (yyyy-mm-dd)

hour

Hour of the day


Submission File Format
sample_submission.csv contains 3 variables - date, hour and demand


Variable

Description

date

Date (yyyy-mm-dd)

hour

Hour of the day

demand

No. of car rentals in a hour


Evaluation metric
The evaluation metric for this hackathon is RMSE score.


Guidelines for Final Submission
Please ensure that your final submission includes the following:

Solution file containing the predictions for the row_id in the test set (Format is given in sample_submission.csv)
A zipped file containing code & approach (Note that both code and approach document are mandatory for shortlisting)
Code: Clean code with comments on each part
Approach: Please share your approach to solve the problem (doc/ppt/pdf format). It should cover the following topics:
A brief on the approach used to solve the problem.
Which Data-preprocessing / Feature Engineering ideas really worked? How did you discover them?
What does your final model look like? How did you reach it?

Public and Private Split
Test data is further divided into Public (40%) and Private (60%) data.

Your initial responses will be checked and scored on the Public data. The final rankings would be based on your private score which will be published once the competition is over.
