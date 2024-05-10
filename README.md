# This project is made for the subject Elements of Data Processing in semester 1 of year 2024.
# Authors: Sonika Agarwal, Jose Augusto Santos, Dorn Kasikumpaiboon

# Objective: 
- Determine and evaluate the popularity of authors based on the quality and quantity of  the ratings that their books receive.
- Develop a recommendation system that accurately recommends books to users and gives a prediction for what they might rate the book.

## Folders in the repo:
- Pre-Processing Folder
- K-means clustering Folder for Objective 1
- Recommendation system folder for Objective 2
- plots for analysis

## Running pre-processing and k-means clustering
- First run book_preprocessing.py in the preprocessing folder
- Then run models.py in the k-means clustering folder
- ** Functions and their respective definitions for both files are in preprocessing_functions.py and model_functions.py respectively

## How to run the files for the recommendation system:
- extract the folder
- check the path to download the data files is correct and working 
- ensure that you have a working setup to run jupyter notebooks
- make sure the libraries are installed properly

## Function and parameters (Recommendation System):
- def RecommendationSystem: This function takes 4 values: The details book you want the prediction of, the dataset of books (pre-processed), the dataset of ratings (pre-processed), and the recommended state as a boolean value
- def evaluate: This function only takes 1 value which is the percentage of the dataset you want to test. We ran it on 0.1% and 1%.

## Recommendation files to run:
- Scripts/RecommendationSystem/RecommendationSystemMain.ipynb

## HISTORY:
- The recommendation system folder contains a folder with the commit history of the system. It also includes instances of count vectorizer within it. 
path : Scripts/RecommendationSystem/RecSystemModel_Evaluation