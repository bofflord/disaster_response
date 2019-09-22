# disaster_response
Udacity Nanodegree Project Disaster Response

# 1. INSTALLATIONS
#### library for processing of command line inputs
import sys
#### import data science libraries
import pandas as pd
import numpy as np
#### for sql lite db
from sqlalchemy import create_engine
#### for pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
#### NLP libraries
import re
import nltk
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
#### library for multi target classification
from sklearn.multioutput import MultiOutputClassifier
#### libraries for estimators
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
#### libraries for NLP feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#### libraries for split in test and training data
from sklearn.model_selection import train_test_split, GridSearchCV
#### libraries for model evaluation
from sklearn.metrics import classification_report
#### pickle library to save machine learning model
import pickle
#### libraries for web app
import json
import plotly
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib

# 2. PROJECT SUMMARY
This project was done as part of the Udacity Data Scientist Program.
This is the corresponding project motivation provided by Udacity:
"In this course, you've learned and built on your data engineering skills to expand your opportunities and potential as a data scientist. In this project, you'll apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In the Project Workspace, you'll find a data set containing real messages that were sent during disaster events. You will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

Your project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off your software skills, including your ability to create basic data pipelines and write clean, organized code!"

The project therefore consists of three main elements:
* Script for data processing
* Script for model training
* Web app that displays training data information and uses to previously trained model to predict on data.


# 3. FILE DESCRIPTIONS
* "process_data.py": python script for data processing
* "train_classifier.py": python script for model training
* "run.py": python script that starts web app
* "master.html": html landing page of web app. Displays the following statistics:
  * tbd
  * tbd
  * tbd
* "go.hmtl": html page that displays classification results of model.
TBD


# 4. OVERVIEW ON REQUIRED FILE STRUCTURE
The above mentioned commands requrire the following file structure:
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model


# 5. HOW TO INTERACT WITH THE PROJECT
TBD

