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



def step_identify_economic(corpus_dir: Path, output_dir: Path):
    """for each xml file on corpus add probability tag that it's economic article."""
    economic_classifier = EconomicClassifier() # load the economic classifier
    
    # Loop through each XML file in the corpus directory
    for xml_file in corpus_dir.glob('*.xml'):
        goid = xml_file.stem
        soup = tdm_parser.get_xml_soup(xml_file)
        value = economic_classifier.predict(corpus_dir)
        soup = tdm_parser.modify_tag(soup, 'is_economic', value)


def step_get_economic_file_names(corpus_dir: Path, prob_threshold: float = 0.5):
    """write a txt file of files that are classified as economic articles."""
    with open(FILE_NAMES_PATH / corpus_dir.stem / 'economic_files.txt', 'w') as f:
        for xml_file in corpus_dir.glob('*.xml'):
            goid = xml_file.stem
            if tdm_parser.get_tag_value(xml_file, 'is_economic') > prob_threshold:
                f.write(f"{goid}\n")
    return None


def step_tfidf_tags(corpus_dir: Path, model_path: Path): #TODO add logging    
    """Append TF-IDF keyword tags to each article."""
    extractor = tf_idf_extractor.TfidfKeywordExtractor(model_path) # load the TF-IDF extractor model
    economic_files = [file_name for file_name in FILE_NAMES_PATH / f"{corpus_dir.stem}/economic_files.txt"] # get economic file names in a list 
    
    # Loop through each XML file in the corpus directory
    for xml_file in corpus_dir.glob('*.xml'):
        if xml_file.stem in economic_files: # only process economic files
            tf_idf_values = extractor.extract_top_keywords(txt_str=tdm_parser.get_article_text(xml_file))
            tdm_parser.modify_tags(xml_file, tf_idf_values, 'tf_idf')


def step_title_sentiment_prob(corpus_dir: Path, model_path: Path, label: str):
    """Add a sentiment label to each article title."""
    analyzer = sentiment_score.TextAnalysis(model_path)
    economic_files = [file_name for file_name in FILE_NAMES_PATH / f"{corpus_dir.stem}/economic_files.txt"] # get economic file names in a list 
    
    # Loop through each XML file in the corpus directory
    for xml_file in corpus_dir.glob('*.xml'):
        if xml_file.stem in economic_files: # only process economic files
            title = tdm_parser.get_article_title(xml_file)
            sentiment_label = analyzer.get_sentiment_prob(title) #TODO
            tdm_parser.modify_tag(xml_file, label, sentiment_label)


def step_title_sentiment_prob(corpus_name: str, output_dir: Path):
    """Store sentiment probabilities for each title."""
    analyzer = sentiment_score.TextAnalysis()
    properties_modifier.modify_csv_title_sentiment_prob(output_dir / corpus_name, corpus_name, analyzer)


def step_paragraph_sentiment_prob(corpus_name: str, output_dir: Path):
    """Calculate sentiment probabilities for article paragraphs."""
    analyzer = sentiment_score.TextAnalysis()
    properties_modifier.modify_csv_paragraph_sentiment_prob(output_dir / corpus_name, corpus_name, analyzer)



    

def step_xml_to_csv(corpus_name: str, output_dir: Path): #TODO fix
    """Convert a corpus of XML files to DataFrame and save as CSV."""
    file_names_path = FILE_NAMES_PATH / corpus_name
    results_path = output_dir / corpus_name
    results_path.mkdir(parents=True, exist_ok=True)
    xml_to_df.xml_to_df(file_names_path, results_path)
    
    
STEP_FUNCTIONS = {
    'economic': step_identify_economic,
    #'train_tfidf': step_train_tfidf,
    #'title_sentiment': step_title_sentiment,
    #'title_sentiment_prob': step_title_sentiment_prob,
    #'paragraph_sentiment_prob': step_paragraph_sentiment_prob,
    #'tfidf_tags': step_tfidf_tags,
}

STEP_DESCRIPTIONS = {
    'economic': 'classify article as economic',
    #'train_tfidf': 'train TF-IDF keyword model',
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
    main()


def train_is_economic(file_names_path: Path, input_dir: Path, output_dir: Path):
    """Train a TF-IDF model on article titles and text."""
    docs = tf_idf_trainer.get_title_body_str(output_dir, corpus_name)
    extractor = tf_idf_extractor.TfidfKeywordExtractor()
    tf_idf_trainer.train_model(docs, extractor, output_dir)
