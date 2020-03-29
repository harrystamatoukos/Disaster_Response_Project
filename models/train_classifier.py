import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import re
from sqlalchemy import create_engine
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, RegexpTokenizer
import sys
import nltk
import numpy as np
from termcolor import colored, cprint
nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath='../data/DisasterResponse.db'):
    '''    A function that loads data from the created database

    INPUT:
        DisasterResponse.db
    OUTPUT:
        X - Returning the messages from the dataset
        y - Returning the categories of the datasets
        category_names - Returning the names of the categories
    '''

    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('StaginMLtable', con=engine)
    print(df.head())
    X = df['message']
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    INPUT:
        The text we need to process

    OUTPUT:
        text after being tokenized, lowercased and lemmatized
    '''

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # take out all punctuation while tokenizing
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # lemmatize as shown in the lesson
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''
    Returns a pipeline model that has undergone tokenization,
    CountVectorization, TfidfTransformation
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [10, 25],
                  'clf__estimator__min_samples_split': [2, 4]}

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    """
    The fucntion prints the precision, recall and f1-score
    """

    y_pred = model.predict(X_test)

    for i, col in enumerate(category_names):
        precision, recall, fscore, support = precision_recall_fscore_support(y_test[col],
                                                                             y_pred[:, i],
                                                                             average='weighted')

        print('\nReport for the column ({}):\n'.format(
            colored(col, 'red', attrs=['bold', 'underline'])))

        if precision >= 0.75:
            print('Precision: {}'.format(colored(round(precision, 2), 'green')))
        else:
            print('Precision: {}'.format(colored(round(precision, 2), 'yellow')))

        if recall >= 0.75:
            print('Recall: {}'.format(colored(round(recall, 2), 'green')))
        else:
            print('Recall: {}'.format(colored(round(recall, 2), 'yellow')))

        if fscore >= 0.75:
            print('F-score: {}'.format(colored(round(fscore, 2), 'green')))
        else:
            print('F-score: {}'.format(colored(round(fscore, 2), 'yellow')))


def save_model(model, model_filepath):
    """ Saves the model to the disk  """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
