"""Command-line pipeline for running article processing steps.

Each function corresponds to a stage in the end-to-end workflow. Steps can be
selected individually using the ``--steps`` argument.
"""

import argparse
from pathlib import Path
import ast
import pandas as pd

from config import (
    FILE_NAMES_PATH,
    RESULTS_PATH,
    xml_to_df,
    properties_modifier,
    sentiment_score,
    tf_idf_trainer,
    tf_idf_extractor,
)
import config as config
from topic_modeling.is_economic_model.train_model import EconomicClassifier


def step_xml_to_df(corpus_name: str, output_dir: Path):
    file_names_path = FILE_NAMES_PATH / corpus_name
    results_path = output_dir / corpus_name
    results_path.mkdir(parents=True, exist_ok=True)
    xml_to_df.xml_to_df(file_names_path, results_path)


def add_xml_tags_to_csvs(corpus_name: str, output_dir: Path, tag_name: str):
    """add XML tags to the csv files. matches the XML files to the CSVs using the file names. compared to GOID column in the CSVs.
    in each csv file there is a column 'goid' which is the GOID of the article. the GOID is the file name of the XML file without the extension.
    This function will add the XML tags to the CSV files by matching the GOID with the file names in the XML directory.
    """
    result_folder_path = output_dir / corpus_name
    #TODO
    
    
    
    #add_a_tdm_tag_to_csv(result_folder_path, corpus_name, tag_name)

def step_economic(corpus_name: str, output_dir: Path):
    """Label each paragraph as economic or not."""
    results_path = output_dir / corpus_name
    classifier = EconomicClassifier()
    for csv_file in sorted(results_path.glob('*.csv')):
        df = pd.read_csv(csv_file)
        df['is_economic'] = df['paragrph_text'].apply(
            lambda t: classifier.is_economic(' '.join(ast.literal_eval(t)) if isinstance(t, str) else '')
        )
        df.to_csv(csv_file, index=False)


def step_train_tfidf(corpus_name: str, output_dir: Path):
    """Train a TF-IDF model on article titles and text."""
    docs = tf_idf_trainer.get_title_body_str(output_dir, corpus_name)
    extractor = tf_idf_extractor.TfidfKeywordExtractor()
    tf_idf_trainer.train_model(docs, extractor, output_dir)


def step_title_sentiment(corpus_name: str, output_dir: Path):
    """Add a sentiment label to each article title."""
    analyzer = sentiment_score.TextAnalysis()
    properties_modifier.modify_csv_title_sentiment(output_dir / corpus_name, corpus_name, analyzer)


def step_title_sentiment_prob(corpus_name: str, output_dir: Path):
    """Store sentiment probabilities for each title."""
    analyzer = sentiment_score.TextAnalysis()
    properties_modifier.modify_csv_title_sentiment_prob(output_dir / corpus_name, corpus_name, analyzer)


def step_paragraph_sentiment_prob(corpus_name: str, output_dir: Path):
    """Calculate sentiment probabilities for article paragraphs."""
    analyzer = sentiment_score.TextAnalysis()
    properties_modifier.modify_csv_paragraph_sentiment_prob(output_dir / corpus_name, corpus_name, analyzer)


def step_tfidf_tags(corpus_name: str, output_dir: Path):
    """Append TF-IDF keyword tags to each article."""
    extractor = tf_idf_extractor.TfidfKeywordExtractor(model_path=output_dir / 'tfidf_vectorizer.pkl')
    properties_modifier.modify_csv_tf_idf(output_dir / corpus_name, corpus_name, extractor, top_n=10)


STEP_FUNCTIONS = {
    'xml_to_df': step_xml_to_df,
    'economic': step_economic,
    'train_tfidf': step_train_tfidf,
    'title_sentiment': step_title_sentiment,
    'title_sentiment_prob': step_title_sentiment_prob,
    'paragraph_sentiment_prob': step_paragraph_sentiment_prob,
    'tfidf_tags': step_tfidf_tags,
}

STEP_DESCRIPTIONS = {
    'xml_to_df': 'convert XML files to CSV',
    'economic': 'classify paragraphs as economic',
    'train_tfidf': 'train TF-IDF keyword model',
    'title_sentiment': 'add sentiment label to titles',
    'title_sentiment_prob': 'compute sentiment probabilities for titles',
    'paragraph_sentiment_prob': 'compute sentiment probabilities for paragraphs',
    'tfidf_tags': 'append top TF-IDF keywords',
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the sentiment pipeline")
    parser.add_argument('--corpus-dir', required=True, help='Name of corpus folder under CORPUSES_PATH')
    parser.add_argument('--output-dir', default=str(RESULTS_PATH), help='Destination folder for processed data')
    parser.add_argument('--steps', default=','.join(STEP_FUNCTIONS.keys()), help='Comma separated list of steps to run')
    return parser.parse_args()


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
