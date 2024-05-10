# This project is made for the subject Elements of Data Processing in semester 1 of year 2024.
## Authors: Sonika Agarwal, Jose Augusto Santos, Dorn Kasikumpaiboon

## Objective: 
- Determine and evaluate the popularity of authors based on the quality and quantity of  the ratings that their books receive.
- Develop a recommendation system that accurately recommends books to users and gives a prediction for what they might rate the book.

## Folders in the repo:
1. SCRIPTS:
- Data Files
- Pre-Processing 
- K-meansClustering (Folder for Objective 1)
- RecommendationSystem (Folder for Objective 2)
2. PLOTS
- plots for analysis

## How to run the files:
- extract the folder
- check the path to download the data files is correct and working 
- ensure that you have a working setup to run jupyter notebooks
- make sure the libraries are installed properly

## Function and parameters:
- def RecommendationSystem: This function takes 4 values: The details book you want the prediction of, the dataset of books (pre-processed), the dataset of ratings (pre-processed), and the recommended state as a boolean value
- def evaluate: This function only takes 1 value which is the percentage of the dataset you want to test. We ran it on 0.1% and 1%.
- ## (SKIP'S functions if any)

## Main files to run:
- ## note to skip: please add if you have any "main" files
- Scripts/RecommendationSystem/RecommendationSystemMain.ipynb

## History:
- The recommendation system folder contains a folder with the commit history of the system. It also includes instances of count vectorizer within it. 
path : Scripts/RecommendationSystem/RecSystemModel_Evaluation

## Data Folder: 
Contains two folders. One with the raw data and the other with the pre-processed data

## Pre-Processing Folder:
Contains files to process age, books dataset and general pre-processing

