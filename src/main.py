# set environment
from pathlib import Path
SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
import sys
sys.path.append(str(SRC_PATH))
from config import *


def corpus_xmls_to_dfs(corpus_name):
    # Build paths for file names and results
    file_names_path = FILE_NAMES_PATH / corpus_name
    results_path = PROJECT_DATA_PATH / 'processed' / 'results' / corpus_name

    # Create the results path if it doesn't exist
    results_path.mkdir(exist_ok=True)

    # Call a function (xml_to_df) that processes the corpus
    xml_to_df(file_names_path, results_path)
    return None

def modify_corpus_title_sentiment(corpus_name):
    result_folder_path = PROJECT_DATA_PATH / 'processed/results/' / corpus_name
    file_names_log_path = LOGS_PATH / corpus_name
    file_names_log_path.mkdir(exist_ok=True)
    modify_csv_title_sentiment(result_folder_path)

def tf_idf_model_trainer(result_folder_path, corpus_name, folder_save_path):
    lst = tf_idf_trainer.get_title_body_str(result_folder_path, corpus_name)
    tf_idf_trainer.train_model(lst, tf_idf_keyword_extractor, folder_save_path)
    return None

def modify_corpus_tf_idf_tags(corpus_name):
    result_folder_path = PROJECT_DATA_PATH / 'processed/results/' / corpus_name
    file_names_log_path = LOGS_PATH / corpus_name
    file_names_log_path.mkdir(exist_ok=True)
    modify_csv_tf_idf(result_folder_path)


def main():
    # Loop over each corpus in CORPUSES_LIST
    for corpus_name in CORPUSES_LIST:
        corpus_xmls_to_dfs(corpus_name)

# If you want this script to be directly runnable:
if __name__ == "__main__":
    main()
    
        
# can run on terminal using the following
#cd /home/ec2-user/SageMaker/david/tdm-sentiment/src/
#python main.py
