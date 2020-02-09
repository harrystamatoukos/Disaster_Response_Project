<h1>Disaster Response Pipeline Project<h1>

<h2>Project Motivation<h2>
<p>In this project, I appled data engineering, natural language processing, and machine learning skills to analyze message data that people sent during disasters to build a model for an API that classifies disaster messages. These messages could potentially be sent to appropriate disaster relief agencies.</p>

<h2>File Descriptions<h2>
  <p>There are three main foleders:</p>

<h4>data</h4>
  <li>  disaster_categories.csv: dataset including all the categories </li>
  <li> disaster_messages.csv: dataset including all the messages </li>
  <li> process_data.py: ETL pipeline scripts to read, clean, and save data into a database </li>
  <li> DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data </li>
<h4>models</h4>
 <li> train_classifier.py: machine learning pipeline scripts to train and export a classifier</li>
 <li> classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer</li>
<h4>app</h4>
 <li> run.py: Flask file to run the web application</li>
templates contains html file for the web applicatin
<h4>Results</h4>
  An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
  A machine learning pipepline was developed to train a classifier to performs multi-output classification on the 36 categories in the dataset.
A Flask app was created to show data visualization and classify the message that user enters on the web page.
<h4>Licensing, Authors, Acknowledgements</h4>
Credits must be given to Udacity for the starter codes and FigureEight for provding the data used by this project.

<h4>Instructions:</h4>
  Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/
