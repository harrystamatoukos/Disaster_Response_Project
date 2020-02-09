Disaster Response Pipeline Project

Project Motivation
In this project, I appled data engineering, natural language processing, and machine learning skills to analyze message data that people sent during disasters to build a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.

File Descriptions
There are three main foleders:

data
  disaster_categories.csv: dataset including all the categories
  disaster_messages.csv: dataset including all the messages
  process_data.py: ETL pipeline scripts to read, clean, and save data into a database
  DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
models
  train_classifier.py: machine learning pipeline scripts to train and export a classifier
  classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
app
  run.py: Flask file to run the web application
templates contains html file for the web applicatin
Results
  An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
  A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
  A Flask app was created to show data visualization and classify the message that user enters on the web page.
Licensing, Authors, Acknowledgements
Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project.

Instructions:
  Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/
