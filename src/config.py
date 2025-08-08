# Project base paths
from pathlib import Path
PROJECT_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/') 
RAW_CORPUSES_PATH = Path('/home/ec2-user/SageMaker/data/')  # e.g., TheWashingtonPostDavid

SRC_PATH = PROJECT_PATH / 'src/'
PROJECT_DATA_PATH = PROJECT_PATH / 'data/'
CORPUSES_PATH = PROJECT_DATA_PATH / 'corpuses/'  # path to store corpus data
RESULTS_PATH = PROJECT_PATH / 'data/processed/results/'  # path to store results
FILE_NAMES_PATH = PROJECT_PATH / 'data/' / 'file_names/'  # Path to store file names
LOGS_PATH = PROJECT_PATH / 'logs/'
CORPUSES_LIST = ['ChicagoTribune', 'LosAngelesTimesDavid', 'Newyork20042023', 'TheJerusalemPost', 'TheWashingtonPostDavid', 'USATodayDavid']  #TODO: List of corpus names
IS_ECONOMIC_MODEL = SRC_PATH / 'topic_modeling/is_economic_model/model/'
TF_IDF_MODEL_PATH = SRC_PATH / 'topic_modeling/tf_idf_model/tfidf_vectorizer.pkl'
BERT_MODEL_PATH = SRC_PATH / 'sentiment/sentiment_model/distilbert-base-uncased-finetuned-sst-2-english/'
ROBERTA_MODEL_PATH = SRC_PATH / 'sentiment/sentiment_model/sentiment-roberta-large-english-3-classes/'

# sys path
import sys
sys.path.append(str(SRC_PATH))

# import custom libraries and functions
import tdm_logger as logger_module 
import data_utils.tdm_parser as tdm_parser_module


import data_utils.file_process as file_process
import data_utils.properties_modifier as properties_modifier

import sentiment.sentiment_model.sentiment_model as sentiment_model

import topic_modeling.is_economic_model.is_economic_model as is_economic_module
import topic_modeling.tf_idf_model.tf_idf_model as tf_idf_extractor
import topic_modeling.tf_idf_model.tf_idf_trainer as tf_idf_trainer
