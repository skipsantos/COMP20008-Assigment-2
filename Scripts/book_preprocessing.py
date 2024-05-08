import pandas as pd
import re
import preprocessing_functions as pf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('data-files/raw-files/BX-Books.csv')

# Filter out all non ASCII characters
df = df[~df['Book-Title'].apply(pf.has_special_characters)]
df = df[~df['Book-Author'].apply(pf.has_special_characters)]
df = df[~df['Book-Publisher'].apply(pf.has_special_characters)]

# Book Title preprocessing steps
# Initialising lemmatizer, stopwords, and tfidf
lemmatizer = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words('english'))

processed_titles = []
for i, title in enumerate(df['Book-Title']):
    processed_titles.append(pf.title_preprocess(title, stop_words,lemmatizer))
df['Title-Tokens'] = [word_tokenize(t) for t in processed_titles]


tfidf_matrix = tfidf_vectorizer.fit_transform(processed_titles)

# Author preprocessing steps
# Setting all author names to lowercase
df['Book-Author'] = df['Book-Author'].apply(lambda x: x.lower())

# Fixing special cases (periods with no spaces)
author_special = df['Book-Author'].apply(lambda x: re.sub(r'\.(?=\w)', '. ', x))

# Removing All punctuations
author_punct = author_special.apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x))

# Abbreviating then joining together single letters into one word (for name initials)
df['Book-Author'] = author_punct.apply(pf.abbreviate)
df['Book-Author'] = df['Book-Author'].apply(pf.process_strings)
df['Author-Tokens'] = df['Book-Author'].apply(word_tokenize)

# Publishing Year Preprocessing Steps
# Convert all years outside plausible range to 0
df.loc[~df['Year-Of-Publication'].between(1920, 2005), 'Year-Of-Publication'] = 0
filtered_df = df[df['Year-Of-Publication'] != 0]

# Get average year of publication for each author
publish_years = filtered_df.groupby('Book-Author')['Year-Of-Publication'].mean().reset_index()

# Replace 0 values with corresponding author mean
df = df.merge(publish_years, on='Book-Author', suffixes=('', '_mean'))
df['Year-Of-Publication'] = df.apply(lambda row: row['Year-Of-Publication_mean']
    if row['Year-Of-Publication'] == 0 else row['Year-Of-Publication'], axis=1)

df['Year-Of-Publication'] = df['Year-Of-Publication'].astype(int)
df.drop(columns='Year-Of-Publication_mean', inplace=True)

# Publisher Preprocessing Steps
processed_publishers = []
for i, title in enumerate(df['Book-Publisher']):
    processed_publishers.append(pf.publisher_preprocess(title, stop_words,lemmatizer))

df['Publisher-Tokens'] = df['Book-Publisher'].apply(word_tokenize)

df.to_csv("data-files/preprocessed-files/Preprocessed_Books.csv")

# Only Uncomment below and run when generating new title tf-idf csv (it's very big!)
# df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# df_tfidf.index = filtered_titles
# df_tfidf.to_csv("data-files/preprocessed-files/Title-Tfidf.csv")