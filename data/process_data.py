import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - to import the database

    OUTPUT

    A merged dataframe of the messages and categories datasets


    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # messages = messages.drop(columns=['original'])

    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    '''
    INPUT
    df : a merged data frame

    OUTPUT
    Retursn a cleaned dataframe

    '''
    # split the categories

    df_categories = df.categories.str.split(pat=';', expand=True)

    # rename the columns with labels from the first row
    df_categories.columns = df_categories.loc[0]

    df_categories.columns = [w[:-2] for w in df_categories.columns]

    for column in df_categories:
        # set each value to be the last character of the string
        df_categories[column] = df_categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        df_categories[column] = pd.to_numeric(df_categories[column])

    df.drop(columns='categories', inplace=True)

    df = pd.concat([df, df_categories], axis=1, join_axes=[df.index])

    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename, table_name='StaginMLtable'):
        """

    INPUT:
        df: Dataframe to be saved
        database_filepath - Filepath used for saving the database
    OUTPUT:
        Saves the database
    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql(table_name, engine, index=False)
    print(f'Data was saved to {database_filename}, in the {table_name} table')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath, table_name='StaginMLtable')

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()

print(len(sys.argv))
