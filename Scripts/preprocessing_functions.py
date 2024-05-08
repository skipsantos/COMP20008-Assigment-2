import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import edit_distance


# Returns True if the string has special characters
def has_special_characters(s):
    return any(ord(char) > 127 for char in s)


def abbreviate(s):
    text = s.split()
    new_string = []
    for i in range(len(text)):
        if i < (len(text) - 1):
            new_string.append(text[i][0])
        else:
            new_string.append(text[i])
    return ' '.join(new_string)


# Joins together consecutive single characters in a string
def process_strings(s):
    text = s.split()
    curr = ''
    new_words = []
    # joining together single letters
    for word in text:
        if len(word) == 1:
            curr += word
        else:
            if curr:
                new_words.append(curr)
                curr = ''
            new_words.append(word)

    if curr:
        new_words.append(curr)

    return ' '.join(new_words)


# Preprocesses titles
def title_preprocess(doc, stop_words, lemmatizer):
    doc = doc.lower()
    # Fixing special cases (periods with no spaces)
    doc_special = re.sub(r'\.(?=\w)', '. ', doc)

    # Removing all punctuation
    doc_punct = re.sub(r'[^A-Za-z0-9\s]', '', doc_special)

    # Remove all instances of 'paperback' in titles
    processed_doc = re.sub(r'paperback\s*', '', doc_punct)

    # Tokenising, removing stop words, then lemmatizing the tokens
    tokens = word_tokenize(processed_doc)
    tokens = [w for w in tokens if not w in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(lemmatized)


# Preprocesses publishers
def publisher_preprocess(doc, stop_words, lemmatizer):
    doc = doc.lower()
    # Fixing special cases (periods with no spaces)
    doc_special = re.sub(r'\.(?=\w)', '. ', doc)

    # Removing all punctuation
    doc_punct = re.sub(r'[^A-Za-z0-9\s]', '', doc_special)

    # Join single characters (initials)
    doc_joined = process_strings(doc_punct)

    # Removing irrelevant words in publishers
    processed_doc = re.sub(r'\b(?:paperback|books|press|publishing|paperbacks)\b', '', doc_joined)

    # Tokenising, removing stop words, then lemmatizing the tokens
    tokens = word_tokenize(processed_doc)
    tokens = [w for w in tokens if not w in stop_words]
    lemmatized = [lemmatizer.lemmatize(w) for w in tokens]

    return ' '.join(lemmatized)
