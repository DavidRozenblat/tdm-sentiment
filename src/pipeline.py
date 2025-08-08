


"""Command-line pipeline for running article processing steps.
Each function corresponds to a stage in the end-to-end workflow. 
"""
from pathlib import Path
from config import *
from bs4 import BeautifulSoup
from collections import deque
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

# Instantiate logger for pipeline steps

tdm_parser = tdm_parser_module.TdmXmlParser()


def run_steps(
    corpus_dir: Path,
    steps: Iterable[Tuple[Callable, Dict[str, Any]]],
    model: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None,
    batch_size: int = 200,
    log_file_name: str = "step_runner",
):
    """Run pipeline ``steps`` on XML files in ``corpus_dir``.

    Parameters
    ----------
    corpus_dir:
        Directory containing XML files.
    steps:
        Iterable of ``(callable, kwargs)`` pairs defining the steps to run.  If a
        step's ``kwargs`` contains ``{"requires_model": True}``, the shared
        ``model`` will be passed to that function under the ``model`` keyword.
    model:
        Optional shared model instance to be used by at most one step.
    context:
        Optional mapping of extra keyword arguments passed to every step.
    batch_size:
        Number of processed files to buffer before updating the log.
    log_file_name:
        Base name for the log file.
    """
    context = context or {}

    requires_model = [kw for _, kw in steps if kw.get("requires_model")]
    if len(requires_model) > 1:
        raise ValueError("Only one step may require a model")
    if requires_model and model is None:
        raise ValueError("A required model was not provided")

    # Determine deterministic file ordering
    initial_files = sorted(xml.name for xml in corpus_dir.glob("*.xml"))
    logger_instance = logger.Logger(
        log_dir=LOGS_PATH,
        log_file_name=log_file_name,
        corpus_name=corpus_dir.stem,
        initiate_file_list=initial_files,
    )
    pending = deque(logger_instance.get_file_names())

    processed_buffer: list[str] = []
    failures: list[Tuple[str, str]] = []
    stats = {
        "total_files": len(initial_files),
        "processed": 0,
        "failed": 0,
    }

    while pending:
        xml_name = pending.popleft()
        xml_path = corpus_dir / xml_name
        try:
            soup = tdm_parser.get_xml_soup(xml_path)
            for func, kw in steps:
                kw = kw.copy()
                if kw.pop("requires_model", False):
                    kw["model"] = model
                kw.update(context)
                soup = func(soup=soup, **kw)
            tdm_parser.write_xml_soup(soup, xml_path)
            stats["processed"] += 1
        except Exception as e:
            failures.append((xml_name, str(e)))
            stats["failed"] += 1
        finally:
            processed_buffer.append(xml_name)
            if len(processed_buffer) >= batch_size:
                logger_instance.update_log_batch(processed_buffer)
                processed_buffer.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)

    return stats, failures




def step_identify_economic(soup: BeautifulSoup, 
                           economic_classifier: is_economic_module.EconomicClassifier): 
    """for each xml file on corpus add probability tag that it's economic article."""
    text = tdm_parser.get_art_text(soup)
    value = economic_classifier.is_economic_prob(text)
    soup = tdm_parser.modify_tag(soup, tag_name='is_economic', value=value, modify=True)
    return soup


def is_above_threshold(soup: BeautifulSoup, prob_threshold: float):
    """write a txt file of files that are classified as economic articles."""
    is_economic = tdm_parser.get_tag_value(soup, 'is_economic')
    # try if is_economic can be converted to numeric
    try:
        is_economic = float(is_economic)
        if is_economic > prob_threshold:
            return True
        else:
            return False
    except:
        return False


def is_economic_step_holder(corpus_dir: Path, del_grades: bool = False, prob_threshold: float = 0.7): 
    """for each xml file on corpus add probability tag that it's economic article."""
    is_economic_classifier = is_economic_module.EconomicClassifier(IS_ECONOMIC_MODEL) # load the economic classifier
    file_path = FILE_NAMES_PATH / corpus_dir.name / 'economic_files.txt'
    file_path.parent.mkdir(parents=True, exist_ok=True) # ensure path exist
    initial_file_list = [xml_file.name for xml_file in corpus_dir.glob('*.xml')] 
    logger_instance = logger.Logger(log_dir=LOGS_PATH, log_file_name = 'is_economic', corpus_name = corpus_dir.stem, initiate_file_list = initial_file_list)
    pending = deque(logger_instance.get_file_names())
    processed_buffer = []

    # Loop through each XML file in the corpus directory
    with open(file_path, 'a') as f:
        while pending:
            xml_name = pending.popleft()
            try:
                xml_path = corpus_dir / xml_name
                goid = xml_path.stem
                soup = tdm_parser.get_xml_soup(xml_path)
                #if del_grades:
                    #tdm_parser.delete_tag(soup, tag_name='grades') 'processed'

                soup = step_identify_economic(soup, is_economic_classifier)
                # rewrite to file
                tdm_parser.write_xml_soup(soup, xml_path)
                # Check if the article is economic and write to file if it is
                if is_above_threshold(soup, prob_threshold):
                    f.write(f"{xml_path.name}\n")
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
            finally:
                # 4) update log regardless of success/failure
                processed_buffer.append(xml_name)
                if len(processed_buffer) >= 200:
                    logger_instance.update_log_batch(processed_buffer)
                    processed_buffer.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)

    print(f"Finished processing {len(initial_file_list)} files. Economic articles saved to {file_path}.")
#---

def step_tfidf_tags(soup: BeautifulSoup, 
                    tfidf_extractor: tf_idf_extractor.TfidfKeywordExtractor,): 
    """Append TF-IDF keyword tags to article."""
    try:
        text = tdm_parser.get_art_text(soup)
        tf_idf_values = tfidf_extractor.extract_top_keywords(txt_str=text)
        soup = tdm_parser.modify_tag(soup, 'tf_idf', tf_idf_values)
    except Exception as e:
        print(f"Error extracting TF-IDF tags: {e}")
    return soup

      
def step_title_sentiment_prob(soup: BeautifulSoup,
                              sentiment_model: sentiment_model.TextAnalysis, 
                              label_dict: dict):
    """Add a sentiment label to each article title."""
    try:
        title = tdm_parser.get_tag_value(soup, 'Title')
        sentiment_dict = sentiment_model.txt_sentiment_dict(title) 
        for label in sentiment_dict.keys():
            soup = tdm_parser.modify_tag(soup, label_dict[label], sentiment_dict[label])  # modify sentiment labels
    except Exception as e:
        print(f"Error processing title sentiment: {e}")
    return soup


def article_average_sentiment_helper(paragraphs, analyzer): 
    """
    Compute the average positive, neutral, and negative scores
    over a list of paragraphs.
    """
    # initialize sums
    sums = {'positive': 0.0, 'neutral': 0.0, 'negative': 0.0}
    if not paragraphs:
        print("Warning: No paragraphs provided for sentiment analysis.")
        return sums
    probs_list = analyzer.batch_txt_sentiment_dict(paragraphs) or []
    n = len(probs_list)

    for probs in probs_list:
        sums['negative'] += probs.get('negative', 0)
        sums['neutral']  += probs.get('neutral',  0)
        sums['positive'] += probs.get('positive', 0)

    if n == 0:
        return {k: 0.0 for k in sums}

    return {k: round(v / n, 4) for k, v in sums.items()}



def step_paragraph_sentiment_prob(soup: BeautifulSoup,
                                  sentiment_model: sentiment_model.TextAnalysis,
                                  label_dict: dict):
    """Calculate sentiment probabilities for article paragraphs."""
    # Loop through each XML file in the corpus directory
    try:
        paragraphs_lst = tdm_parser.get_art_text(soup, return_str=False)
        sentiment_dict = article_average_sentiment_helper(paragraphs_lst, sentiment_model)
        for label in sentiment_dict.keys():
            soup = tdm_parser.modify_tag(soup, label_dict[label], sentiment_dict[label])  # modify sentiment labels
    except Exception as e:
        print(f"Error processing paragraph sentiment: {e}")
    return soup

    

def main_step_holder(corpus_dir: Path,
                    log_file_name: str,
                    roberta_title_sentiment_label_dict: dict,
                    roberta_paragraph_sentiment_label_dict: dict,
                    bert_title_sentiment_label_dict: dict,
                    bert_paragraph_sentiment_label_dict: dict):
    """Main step holder for processing a corpus directory."""
    tfidf_extractor = tf_idf_extractor.TfidfKeywordExtractor(TF_IDF_MODEL_PATH)

    # get the list of economic files and initialize logger
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger.Logger(
        log_dir=LOGS_PATH,
        log_file_name=log_file_name,
        corpus_name=corpus_dir.stem,
        initiate_file_list=economic_files,
    )

    # First pass: process TF-IDF and RoBERTa sentiments
    pending = deque(economic_files)
    with sentiment_model.TextAnalysis(ROBERTA_MODEL_PATH) as roberta_sentiment_analyzer:
        while pending:
            xml_name = pending.popleft()
            try:
                xml_file = corpus_dir / xml_name
                soup = tdm_parser.get_xml_soup(xml_file)
                soup = step_tfidf_tags(soup=soup, tfidf_extractor=tfidf_extractor)
                soup = step_title_sentiment_prob(
                    soup=soup,
                    sentiment_model=roberta_sentiment_analyzer,
                    label_dict=roberta_title_sentiment_label_dict,
                )
                soup = step_paragraph_sentiment_prob(
                    soup=soup,
                    sentiment_model=roberta_sentiment_analyzer,
                    label_dict=roberta_paragraph_sentiment_label_dict,
                )
                tdm_parser.write_xml_soup(soup, xml_file)
            except Exception as e:
                print(f"Error processing {xml_name}: {e}")

    # Second pass: load BERT model and append its sentiment scores
    pending = deque(economic_files)
    with sentiment_model.TextAnalysis(BERT_MODEL_PATH) as bert_sentiment_analyzer:
        while pending:
            xml_name = pending.popleft()
            try:
                xml_file = corpus_dir / xml_name
                soup = tdm_parser.get_xml_soup(xml_file)
                soup = step_title_sentiment_prob(
                    soup=soup,
                    sentiment_model=bert_sentiment_analyzer,
                    label_dict=bert_title_sentiment_label_dict,
                )
                soup = step_paragraph_sentiment_prob(
                    soup=soup,
                    sentiment_model=bert_sentiment_analyzer,
                    label_dict=bert_paragraph_sentiment_label_dict,
                )
                tdm_parser.write_xml_soup(soup, xml_file)
            except Exception as e:
                print(f"Error processing {xml_name}: {e}")
            finally:
                logger_instance.update_log_file(xml_name)




def step_xml_to_csv(corpus_dir: Path, output_dir: Path): 
    """Convert a corpus of XML files to DataFrame and save as CSV."""
    pass
    
STEP_FUNCTIONS = {
    #'economic': step_identify_economic,
    #'title_sentiment': step_title_sentiment,
    #'title_sentiment_prob': step_title_sentiment_prob,
    #'paragraph_sentiment_prob': step_paragraph_sentiment_prob,
    #'tfidf_tags': step_tfidf_tags,
}

STEP_DESCRIPTIONS = {
    #'economic': 'classify article as economic',
    #'title_sentiment': 'add sentiment label to titles',
    #'title_sentiment_prob': 'compute sentiment probabilities for titles',
    #'paragraph_sentiment_prob': 'compute sentiment probabilities for paragraphs',
    #'tfidf_tags': 'append top TF-IDF keywords',
}



def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run pipeline steps on a corpus")
    parser.add_argument("corpus_dir", type=Path, help="Directory containing XML files")
    parser.add_argument("steps", nargs="+", choices=STEP_FUNCTIONS.keys(), help="Steps to execute")
    parser.add_argument("--batch-size", type=int, default=200, dest="batch_size")
    parser.add_argument("--log-file-name", default="step_runner", dest="log_file_name")
    args = parser.parse_args()

    step_defs = [(STEP_FUNCTIONS[name], {}) for name in args.steps]
    run_steps(
        corpus_dir=args.corpus_dir,
        steps=step_defs,
        batch_size=args.batch_size,
        log_file_name=args.log_file_name,
    )


if __name__ == "__main__":
    main()
