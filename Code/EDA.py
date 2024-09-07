import re
import numpy as np
import pandas as pd

import matplotlib.cm as cm
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import word_tokenize

import clean_reports
import preprocess_reports

from sentence_transformers import SentenceTransformer

nltk.download("punkt")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# dataset location
DATASET = "Data/prospect-data.csv"

# load dataset into dataframe
data = clean_reports.clean(DATASET, raw=True)

data.head()