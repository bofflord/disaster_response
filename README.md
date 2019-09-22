# disaster_response
Udacity Nanodegree Project Disaster Response

# 1. INSTALLATIONS
# library for processing of command line inputs
import sys
# import data science libraries
import pandas as pd
import numpy as np
# import libraries
import pandas as pd
import numpy as np
# for sql lite db
from sqlalchemy import create_engine
# for pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
# NLP libraries
import re
import nltk
nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
# library for multi target classification
from sklearn.multioutput import MultiOutputClassifier
# libraries for estimators
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
# libraries for NLP feature extraction
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# libraries for split in test and training data
from sklearn.model_selection import train_test_split, GridSearchCV
# libraries for model evaluation
from sklearn.metrics import classification_report
# pickle library to save machine learning model
import pickle
# libraries for web app
import json
import plotly
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib

# 2. PROJECT SUMMARY
TBD


# 3. FILE DESCRIPTIONS
TBD


# 4. OVERVIEW ON REQUIRED FILE STRUCTURE
TBD


# 5. HOW TO INTERACT WITH THE PROJECT
TBD

