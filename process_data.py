import sys
# import data science libraries
import pandas as pd
import numpy as np

# for sql lite db
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load messages and categories csv data into pandas dataframes.
    
    ARGS:
    messages_filepath: path to messages csv file
    categories_filepath: path to catgeories csv file
    
    OUTPUT:
    df: merged dataframe of messages and categories
    
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, on="id")
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(pat=";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    category_colnames = row.apply(lambda col: col[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda val: val[-1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    # drop the original categories column from `df`
    df=df.drop("categories", axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis = 1)
    return df


def clean_data(df):
    '''
    Function for data cleaning.
    
    ARGS:
    df: uncleaned dataframe
    
    OUTPUT:
    df: cleaned dataframe
    
    '''
    # drop duplicates
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    '''
    Function for data cleaning.
    
    ARGS:
    df: dataframe
    database_filename: filename of sql lite database
    
    OUTPUT:
    no output
    
    '''
    #engine = create_engine(database_filename)
    engine = create_engine('sqlite:////home/workspace/data/' + database_filename)
    df.to_sql('InsertTableName', engine, index=False)  
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()