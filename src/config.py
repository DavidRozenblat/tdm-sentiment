# Project base paths
from pathlib import Path
PROJECT_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/') 
CORPUSES_PATH = Path('/home/ec2-user/SageMaker/data/')  # e.g., TheWashingtonPostDavid

SRC_PATH = PROJECT_PATH / 'src/'
PROJECT_DATA_PATH = PROJECT_PATH / 'data/'
RESULTS_PATH = PROJECT_PATH / 'data/processed/results/'  # path to store results
FILE_NAMES_PATH = PROJECT_PATH / 'data/' / 'file_names/'  # Path to store file names
LOGS_PATH = PROJECT_PATH / 'logs/'

IS_ECONOMIC_MODEL = SRC_PATH / 'topic_modeling/is_economic_model/model/'
TF_IDF_MODEL_PATH = SRC_PATH / 'topic_modeling/tf_idf_model/tfidf_vectorizer.pkl'
BERT_MODEL_PATH = SRC_PATH / 'sentiment/sentiment_model/distilbert-base-uncased-finetuned-sst-2-english/'

# sys path
import sys
sys.path.append(str(SRC_PATH))

# import custom libraries and functions
import logger as logger 
import data_utils.tdm_parser as tdm_parser

import data_utils.xml_to_df as xml_to_df
import data_utils.properties_modifier as properties_modifier

import sentiment.sentiment_model.sentiment_score as sentiment_score

import topic_modeling.tf_idf_model.keyword_extractor as tf_idf_extractor
import topic_modeling.tf_idf_model.tf_idf_trainer as tf_idf_trainer
