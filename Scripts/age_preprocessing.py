import pandas as pd
import numpy as np
import re
def preprocessedAge():
    # Load data
    books_df = pd.read_csv("Data-Files/BX-Books.csv")
    users_df = pd.read_csv("Data-Files/BX-Users.csv")
    ratings_df = pd.read_csv("Data-Files/BX-Ratings.csv")

    # Clean user age data
    users_df["User-Age"] = users_df["User-Age"].apply(onlyNumber)

    # Merge dataframes
    ratings_users_df = pd.merge(ratings_df, users_df, on="User-ID")
    ratings_users_books_df = pd.merge(ratings_users_df, books_df, on="ISBN")

    # Calculate average user age per book
    avg_user_age_per_book = ratings_users_books_df.groupby("ISBN")["User-Age"].mean().reset_index()
    avg_user_age_per_book.rename(columns={'User-Age': 'Avg-User-Age'}, inplace=True)

    # Merge the average user age back to the main dataframe
    combined_df = pd.merge(ratings_users_books_df, avg_user_age_per_book, on="ISBN")

    # Calculate average age based on book for each user
    avg_book_age_per_user = combined_df.groupby("User-ID")["Avg-User-Age"].mean().reset_index()
    avg_book_age_per_user.rename(columns={'Avg-User-Age': 'Avg-Book-Age'}, inplace=True)

    # Merge this average back to the main dataframe
    final_df = pd.merge(combined_df, avg_book_age_per_user, on="User-ID")

    # Use user age when available, if not use average book age
    final_df["Adjusted-User-Age"] = np.where(final_df["User-Age"].notna(),
                                             final_df["User-Age"],
                                             final_df["Avg-Book-Age"])

    # need more preprocessing from address have 121 nan to be process by address? should I just delete?
    #based_by_loc = userDF.groupby("ISBN")["User-Age"].mean()
    #print(based_by_books.isna().sum()) #74 NA

    #return final_df.isna().sum()
    return final_df

def onlyNumber(age):
    try:
        string_age = str(age)
        cleaned_age = int(re.sub(r'[^0-9]', '', string_age))
        if cleaned_age<0 or cleaned_age> 123: #oldest person in the world is 122 years old
            return np.nan
        return cleaned_age
    except:
        return np.nan

print(preprocessedAge())
