import pandas as pd
import re
import preprocessing_functions as pf
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('Data-Files/Raw-Files/BX-Books.csv')
# Filter out all non ASCII characters
filtered_titles = df[~df['Book-Title'].apply(pf.has_special_characters)]['Book-Title']
filtered_authors = df[~df['Book-Author'].apply(pf.has_special_characters)]['Book-Author']

# Author preprocessing steps
# Setting all author names to lowercase
df['Book-Author'] = df['Book-Author'].apply(lambda x: x.lower())

# Fixing special cases (periods with no spaces)
author_special = filtered_authors.apply(lambda x: re.sub(r'\.(?=\w)', '. ', x))

# Removing All punctuations
author_punct = author_special.apply(lambda x: re.sub(r'[^A-Za-z\s]', '', x))

# Joining together single letters into one word (for name initials)
processed_authors = author_punct.apply(pf.join_characters)
df['Author-Tokens'] = processed_authors.apply(word_tokenize)

# Book Title preprocessing steps
# Initialising lemmatizer, stopwords, and tfidf
lemmatizer = WordNetLemmatizer()
tfidf_vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words('english'))

processed_titles = []
for i, title in enumerate(filtered_titles):
    processed_titles.append(pf.text_preprocess(title, stop_words,lemmatizer))

tfidf_matrix = tfidf_vectorizer.fit_transform(processed_titles)

# Only Uncomment below and run when generating new title tf-idf csv (it's very big!)
# df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# df_tfidf.index = filtered_titles
# df_tfidf.to_csv("Data-Files/Preprocessed-Files/Title-Tfidf.csv")

#df.to_csv('Data-Files/Preprocessed-Files/Updated_Books.csv', index=False)
