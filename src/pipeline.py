"""Command-line pipeline for running article processing steps.

Each function corresponds to a stage in the end-to-end workflow. Steps can be
selected individually using the ``--steps`` argument.
"""
import argparse
from pathlib import Path
import ast
import pandas as pd
from config import *
from topic_modeling.is_economic_model.train_model import EconomicClassifier

# Instantiate logger for pipeline steps

parser = tdm_parser.TdmXmlParser()


def step_identify_economic(corpus_dir: Path):
    """for each xml file on corpus add probability tag that it's economic article."""
    economic_classifier = EconomicClassifier() # load the economic classifier
    
    # Loop through each XML file in the corpus directory
    for xml_path in corpus_dir.glob('*.xml'):
        goid = xml_path.stem
        text = parser.get_art_text(xml_path)
        value = economic_classifier.is_economic_prob(text)
        parser.modify_tag(xml_path, tag_name='is_economic', value=value, modify=True)
        

def step_write_economic_file_names(corpus_dir: Path, prob_threshold: float = 0.7):
    """write a txt file of files that are classified as economic articles."""
    file_path = FILE_NAMES_PATH / corpus_dir.name / 'economic_files.txt'
    file_path.parent.mkdir(parents=True, exist_ok=True) # ensure path exist
    with open(file_path, 'w') as f:
        for xml_file in corpus_dir.glob('*.xml'):
            goid = xml_file.stem
            is_economic = parser.get_tag_value(xml_file, 'is_economic')
            # try if is_economic can be converted to numeric
            try:
                is_economic = float(is_economic)
                if is_economic > prob_threshold:
                    f.write(f"{xml_file.name}\n")
            except:
                continue
    return None


def step_tfidf_tags(corpus_dir: Path, model_path: Path): 
    """Append TF-IDF keyword tags to each article."""
    extractor = tf_idf_extractor.TfidfKeywordExtractor(model_path)  # load the TF-IDF extractor model
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    
    logger_instance = logger.Logger(log_dir=LOGS_PATH, log_file_name = 'tfidf_tags', corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    log_list = logger_instance.get_file_names()

    for xml_name in list(log_list):
        xml_file = corpus_dir / xml_name
        text = parser.get_art_text(xml_file)
        tf_idf_values = extractor.extract_top_keywords(txt_str=text)
        parser.modify_tag(xml_file, 'tf_idf', tf_idf_values)
        log_list.remove(xml_name)
        logger_instance.update_log_file(log_list)



def title_sentiment_probs(text, sentiment_analyzer):
    """
    Get the sentiment probabilities of a single title text.
    """
    # Note: The function expects a list of texts; we wrap text in a list.
    # Truncate the text if it exceeds 512 characters (or tokens as needed)
    if len(text) > 512:
        print(f'text Truncate, len text is{len(text)}')
        text = text[:511]
    try:
        return sentiment_analyzer.get_sentiment_dict(text)
    except Exception as e:
        print(f'error: {e}, input was:{text}')
        return None

    
    
def step_title_sentiment_prob(corpus_dir: Path, model_path: Path, label_dict: dict):
    """Add a sentiment label to each article title."""
    analyzer = sentiment_score.TextAnalysis(model_path)
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    
    logger_instance = logger.Logger(log_dir=LOGS_PATH, log_file_name = 'roberta_sentiment', corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    log_list = logger_instance.get_file_names()
    
    
    # Loop through each XML file in the corpus directory
    for xml_name in list(log_list):
        xml_file = corpus_dir / xml_name
        title = parser.get_article_title(xml_file)
        sentiment_dict = analyzer.title_sentiment_probs(title, analyzer) #TODO
        tdm_parser.modify_tag(xml_file, label_dict['negative'], sentiment_dict['negative']) # modify_negative
        tdm_parser.modify_tag(xml_file, label_dict['netural'], sentiment_dict['netural']) # modify_netural
        tdm_parser.modify_tag(xml_file, label_dict['positive'], sentiment_dict['positive']) # modify_positive
    
        log_list.remove(xml_name)
        logger_instance.update_log_file(log_list)

def step_title_sentiment_prob(corpus_dir: Path):
    """Store sentiment probabilities for each title."""
    pass

def step_paragraph_sentiment_prob(corpus_dir: Path):
    """Calculate sentiment probabilities for article paragraphs."""
    pass

    

def step_xml_to_csv(corpus_dir: Path, output_dir: Path): 
    """Convert a corpus of XML files to DataFrame and save as CSV."""
    pass
    
STEP_FUNCTIONS = {
    'economic': step_identify_economic,
    #'title_sentiment': step_title_sentiment,
    #'title_sentiment_prob': step_title_sentiment_prob,
    #'paragraph_sentiment_prob': step_paragraph_sentiment_prob,
    #'tfidf_tags': step_tfidf_tags,
}

STEP_DESCRIPTIONS = {
    'economic': 'classify article as economic',
    #'title_sentiment': 'add sentiment label to titles',
    #'title_sentiment_prob': 'compute sentiment probabilities for titles',
    #'paragraph_sentiment_prob': 'compute sentiment probabilities for paragraphs',
    #'tfidf_tags': 'append top TF-IDF keywords',
}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the sentiment pipeline")
    parser.add_argument('--corpus-dir', required=True, help='Name of corpus folder under CORPUSES_PATH')
    parser.add_argument('--output-dir', default=str(RESULTS_PATH), help='Destination folder for processed data')
    steps_list = ', '.join(f"{k}: {v}" for k, v in STEP_DESCRIPTIONS.items())
    parser.add_argument(
        '--steps',
        default=','.join(STEP_FUNCTIONS.keys()),
        help='Comma separated list of steps to run. Available steps: ' + steps_list
    )

def main():
    args = parse_args()
    corpus_name = Path(args.corpus_dir).name
    output_dir = Path(args.output_dir)

    steps = [s.strip() for s in args.steps.split(',') if s.strip()]
    for step in steps:
        func = STEP_FUNCTIONS.get(step)
        if not func:
            print(f"Unknown step '{step}', skipping")
            continue
        print(f"\n=== Running step: {step} ===")
        func(corpus_name, output_dir)


if __name__ == '__main__':
    #main()
    corpus_dir = CORPUSES_PATH / 'LosAngelesTimesDavid'
    #step_write_economic_file_names(corpus_dir, prob_threshold=0.2)
    #step_identify_economic(corpus_dir)
    step_tfidf_tags(corpus_dir=corpus_dir, model_path=TF_IDF_MODEL_PATH)

