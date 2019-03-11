import sys
import os
import pandas as pd
import pickle
import numpy as np
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
        
    """
    Description:
        Loads dataset from database
    
    Parameters:
        database_filepath (string): Filepath to SQLite database file
    
    Returns:
        (DataFrame) X: Independent Variables
        (DataFrame) y: Dependent Variables
        (DataFrame) category_names: Data Column Labels
    """

    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_query('SELECT * FROM tbl_messages', engine)
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names = y.columns
    return X, y, category_names

def tokenize(text):
            
    """
    Description:
        Tokenizes message data
    
    Parameters:
       text (string): message text data
    
    Returns:
        (DataFrame) clean_messages: array of tokenized message data
    """

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_messages = []
    for tok in tokens:
        clean_mess = lemmatizer.lemmatize(tok).lower().strip()
        clean_messages.append(clean_mess)
    
    return clean_messages

def build_model():
            
    """
    Description:
        Builds machine learning pipeline
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=1)))
    ])
    
    parameters = {
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__n_estimators': [5, 10, 20],
        'clf__estimator__min_samples_split': [2, 4, 6]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3, n_jobs=-2)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
                
    """
    Description:
        Evaluates models performance in predicting message categories
    
    Parameters:
        model (Classification): stored classification model
        X_test (string): Independent Variables
        Y_test (string): Dependent Variables
        category_names (DataFrame): Stores message category labels
    """
    
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))

def save_model(model, model_filepath):
                
    """
    Description:
        Saves trained classification model to pickle file
    
    Parameters:
        model (Classification): stored classification model
        model_filepath (string): Filepath to pickle file
    """
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()