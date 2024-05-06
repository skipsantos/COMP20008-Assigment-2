import pandas as pd
import re
import preprocessing_functions as pf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

ratings_df = pd.read_csv('Data-Files/Raw-Files/BX-Ratings.csv')
books_df = pd.read_csv('Data-Files/Preprocessed-Files/Preprocessed_Books.csv')

merged_df = ratings_df.merge(books_df, on='ISBN')

author_ratings = merged_df.groupby('Book-Author')['Book-Rating'].mean().reset_index()
year_ratings = average_ratings = merged_df.groupby('Year-Of-Publication')['Book-Rating'].mean().reset_index()


print(year_ratings)
print(author_ratings)

