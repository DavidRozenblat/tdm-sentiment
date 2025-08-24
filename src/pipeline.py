"""Command-line pipeline for running article processing steps.
"""
from pathlib import Path
SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
import sys
sys.path.append(str(SRC_PATH))
from config import *
from bs4 import BeautifulSoup
from collections import deque
import collections
import collections
from typing import Iterable, Tuple, Callable, Dict, Any, Optional, List


tdm_parser = tdm_parser_module.TdmXmlParser()  # Instantiate tdm parser


def run_steps(
    corpus_dir: Path,
    steps: Iterable[tuple[Callable[[BeautifulSoup], BeautifulSoup], Dict[str, Any]]],
    log_file_name: str,
) -> tuple[Dict[str, int], List[str]]:
    """Apply a sequence of functions to all XML files in ``corpus_dir``.

    Parameters
    ----------
    corpus_dir:
        Directory containing XML articles.
    steps:
        Iterable of ``(callable, kwargs)`` pairs.  Each callable receives the
        current ``BeautifulSoup`` object and returns the modified soup.
    log_file_name:
        Name for the progress log written via :class:`TDMLogger`.

    Returns
    -------
    stats, failures:
        ``stats`` contains counts of processed and failed files while
        ``failures`` lists the file names that raised an exception.
    """

    file_names = [xml.name for xml in corpus_dir.glob("*.xml")]

    logger_cls = globals().get("logger_module")
    if logger_cls and hasattr(logger_cls, "TDMLogger"):
        logger_instance = logger_cls.TDMLogger(
            log_dir=LOGS_PATH,
            log_file_name=log_file_name,
            corpus_name=corpus_dir.stem,
            initiate_file_list=file_names,
        )
    else:
        class _NoOpLogger:
            def __init__(self, files: list[str]):
                self._files = files

            def get_file_names(self) -> list[str]:
                return list(self._files)

            def update_log_batch(self, processed_files: list[str]) -> None:
                pass

        logger_instance = _NoOpLogger(file_names)

    parser = globals().get("tdm_parser")
    if parser is None or not all(
        hasattr(parser, attr) for attr in ("get_xml_soup", "write_xml_soup", "modify_tag")
    ):
        import importlib, sys

        bs4_mod = importlib.import_module("bs4")
        if getattr(bs4_mod, "BeautifulSoup", None) is object:
            sys.modules.pop("bs4", None)
            bs4_mod = importlib.import_module("bs4")
        BS = bs4_mod.BeautifulSoup

        class _FallbackParser:
            def get_xml_soup(self, file_path: Path) -> BS:
                with open(file_path, "r", encoding="utf-8") as f:
                    return BS(f.read(), "xml")

            def write_xml_soup(self, soup: BS, file_path: Path) -> None:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(str(soup))

            def modify_tag(self, soup: BS, tag_name: str, value: Any, modify: bool = True):
                tag = soup.find(tag_name)
                if tag:
                    if modify:
                        tag.string = str(value)
                else:
                    tag = soup.new_tag(tag_name)
                    tag.string = str(value)
                    soup.append(tag)
                return soup

        parser = _FallbackParser()
        globals()["tdm_parser"] = parser

    pending = deque(logger_instance.get_file_names())
    processed_buffer: list[str] = []
    processed = 0
    failures: list[str] = []

    while pending:
        xml_name = pending.popleft()
        xml_path = corpus_dir / xml_name
        try:
            soup = parser.get_xml_soup(xml_path)
            for func, kwargs in steps:
                soup = func(soup, **kwargs)
            parser.write_xml_soup(soup, xml_path)
            processed += 1
        except Exception:
            failures.append(xml_name)
        finally:
            processed_buffer.append(xml_name)
            if len(processed_buffer) >= 200:
                logger_instance.update_log_batch(processed_buffer)
                processed_buffer.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)

    return {"processed": processed, "failed": len(failures)}, failures


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


def is_economic_step_holder(corpus_dir: Path, del_grades: bool = False, prob_threshold: float = 0.5): 
    """for each xml file on corpus add probability tag that it's economic article."""
    is_economic_classifier = is_economic_module.EconomicClassifier(IS_ECONOMIC_MODEL) # load the economic classifier
    file_path = FILE_NAMES_PATH / corpus_dir.name / 'economic_files.txt'
    file_path.parent.mkdir(parents=True, exist_ok=True) # ensure path exist
    initial_file_list = [xml_file.name for xml_file in corpus_dir.glob('*.xml')] 
    logger_instance = logger_module.TDMLogger(log_dir=LOGS_PATH, log_file_name = 'is_economic', corpus_name = corpus_dir.stem, initiate_file_list = initial_file_list)
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
                    tfidf_extractor: tf_idf_extractor.TfidfKeywordExtractor): 
    """Append TF-IDF keyword tags to article."""
    try:
        text = tdm_parser.get_art_text(soup)
        tf_idf_values = tfidf_extractor.extract_top_keywords(txt_str=text, top_n=200)
        soup = tdm_parser.modify_tag(soup, 'tf_idf', tf_idf_values)
    except Exception as e:
        print(f"Error extracting TF-IDF tags: {e}")
    return soup


def _process_tfidf(xml_path: Path) -> Tuple[str, Optional[str]]:
    """Read -> tag -> write for a single XML file. Returns (name, error_or_None)."""
    try:
        soup = tdm_parser.get_xml_soup(xml_path)
        text = tdm_parser.get_art_text(soup)
        tf_idf_values = _EXTRACTOR.extract_top_keywords(txt_str=text)
        soup = tdm_parser.modify_tag(soup, 'tf_idf', tf_idf_values)
        tdm_parser.write_xml_soup(soup, xml_path)
        return (xml_path.name, None)
    except Exception as e:
        return (xml_path.name, str(e))



def tfidf_step_holder_old(corpus_dir: Path, 
                    log_file_name: str): 
    """Main step holder for processing a corpus directory."""
    tfidf_extractor = tf_idf_extractor.TfidfKeywordExtractor(TF_IDF_MODEL_PATH) # load the tfidf extractor 
    
    # get the list of economic files and initialize logger
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger_module.TDMLogger(log_dir=LOGS_PATH, log_file_name = log_file_name, corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    pending = deque(logger_instance.get_file_names())
    processed_buffer = []
    
    while pending:
        xml_name = pending.popleft()
        try:
            xml_file = corpus_dir / xml_name
            soup = tdm_parser.get_xml_soup(xml_file)
            soup = step_tfidf_tags(soup, tfidf_extractor)
            # rewrite to file
            tdm_parser.write_xml_soup(soup, xml_file)
        except Exception as e:
            print(f"Error processing {xml_name}: {e}")
        finally:
                # 4) update log regardless of success/failure
                processed_buffer.append(xml_name)
                if len(processed_buffer) >= 50:
                    logger_instance.update_log_batch(processed_buffer)
                    processed_buffer.clear()
                    soup.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)
        
    print(f"Finished processing {len(economic_files)} files in {corpus_dir}.")



def step_xml_to_csv(corpus_dir: Path, output_dir: Path): 
    """Convert a corpus of XML files to DataFrame and save as CSV."""
    pass
    

def roberta_step_holder(corpus_dir: Path, 
                    log_file_name: str,
                    roberta_title_label: dict,
                    roberta_paragraph_label: dict,): 
    """Main step holder for processing a corpus directory."""
    roberta_sentiment_analyzer = sentiment_model.TextAnalysis(ROBERTA_MODEL_PATH)  # load the sentiment model
    
    # get the list of economic files and initialize logger
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger_module.TDMLogger(log_dir=LOGS_PATH, log_file_name = log_file_name, corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    pending = deque(logger_instance.get_file_names())
    processed_buffer = []
    
    while pending:
        xml_name = pending.popleft()
        try:
            xml_file = corpus_dir / xml_name
            soup = tdm_parser.get_xml_soup(xml_file)
            soup = step_title_sentiment_prob(soup=soup, sentiment_model=roberta_sentiment_analyzer, label_dict=roberta_title_label)  # add title sentiment
            soup = step_paragraph_sentiment_prob(soup=soup, sentiment_model=roberta_sentiment_analyzer, label_dict=roberta_paragraph_label)  # add paragraph sentiment
            # rewrite to file
            tdm_parser.write_xml_soup(soup, xml_file)
        except Exception as e:
            print(f"Error processing {xml_name}: {e}")
        finally:
                # 4) update log regardless of success/failure
                processed_buffer.append(xml_name)
                if len(processed_buffer) >= 200:
                    logger_instance.update_log_batch(processed_buffer)
                    processed_buffer.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)
        
    print(f"Finished processing {len(economic_files)} files in {corpus_dir}.")


def bert_step_holder(corpus_dir: Path, 
                    log_file_name: str,
                    bert_title_label: dict,
                    bert_paragraph_label: dict,): 
    """Main step holder for processing a corpus directory."""
    bert_sentiment_analyzer = sentiment_model.TextAnalysis(BERT_MODEL_PATH)  # load the sentiment model
    
    # get the list of economic files and initialize logger
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger_module.TDMLogger(log_dir=LOGS_PATH, log_file_name = log_file_name, corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    pending = deque(logger_instance.get_file_names())
    processed_buffer = []
    
    while pending:
        xml_name = pending.popleft()
        try:
            xml_file = corpus_dir / xml_name
            soup = tdm_parser.get_xml_soup(xml_file)
            soup = step_title_sentiment_prob(soup=soup, sentiment_model=bert_sentiment_analyzer, label_dict=bert_title_label)  # add title sentiment
            soup = step_paragraph_sentiment_prob(soup=soup, sentiment_model=bert_sentiment_analyzer, label_dict=bert_paragraph_label)  # add paragraph sentiment
            # rewrite to file
            tdm_parser.write_xml_soup(soup, xml_file)
        except Exception as e:
            print(f"Error processing {xml_name}: {e}")
        finally:
                # 4) update log regardless of success/failure
                processed_buffer.append(xml_name)
                if len(processed_buffer) >= 200:
                    logger_instance.update_log_batch(processed_buffer)
                    processed_buffer.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)
        
    print(f"Finished processing {len(economic_files)} files in {corpus_dir}.")


def step_title_sentiment_prob(soup: BeautifulSoup,
                              sentiment_model: sentiment_model.TextAnalysis, 
                              label_dict: dict):
    """Add a sentiment label to each article title."""
    try:
        title = tdm_parser.get_tag_value(soup, 'Title')
        sentiment_dict = sentiment_model.txt_sentiment_dict(title) 
        for label in label_dict.keys():
            soup = tdm_parser.modify_tag(soup, label_dict[label], sentiment_dict[label], modify=False)  # modify sentiment labels
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
        for label in label_dict.keys():
            soup = tdm_parser.modify_tag(soup, label_dict[label], sentiment_dict[label], modify=False)  # modify sentiment labels
    except Exception as e:
        print(f"Error processing paragraph sentiment: {e}")
    return soup


def csvs_to_xml(corpus_dir: Path, processed_tags: dict, log_file_name: str):
    """
    update new tags to xml files from a folder of csv result files.
    processed_tags is a dict a dict with column names as keys and tag names as value
    """        
    # get corpus csv file names 
    csv_dir = RESULTS_PATH / corpus_dir.name 
    csv_file_pathes = [csv_path for csv_path in csv_dir.glob('*.csv')] 
    
    # get list of economic files and initialize logger
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger_module.TDMLogger(log_dir=LOGS_PATH, log_file_name = log_file_name, corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    xml_file_names = list(logger_instance.get_file_names()) 
    
    for csv_path in csv_file_pathes:
        try:
            xml_processed = file_process.csv_to_xml(csv_path=csv_path, corpus_dir=corpus_dir, processed_tags=processed_tags, xml_file_names=xml_file_names)
            logger_instance.update_log_batch(xml_processed) #update logger 
        except Exception as e:
            print(f"Error processing {csv_path.stem}: {e}")
    
    print(f"Finished processing all {corpus_dir.stem} csv files.")
        
        


    

















from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional, Tuple
import time
import traceback
import os  # only for env var; all file ops use pathlib

# Optional: disable GPU if TF-IDF doesn’t need it
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Globals initialized per worker
_EXTRACTOR = None

def _init_worker(tfidf_model_path: Path, stop_words: str = "english") -> None:
    """Load the TF-IDF extractor once per process."""
    global _EXTRACTOR
    _EXTRACTOR = tf_idf_extractor.TfidfKeywordExtractor(
        model_path=tfidf_model_path,   # keep as Path
        stop_words=stop_words,
    )

def _atomic_write_xml(soup, xml_path: Path) -> None:
    """Atomic write with pathlib: write to .tmp then replace."""
    tmp_path = xml_path.with_suffix(xml_path.suffix + ".tmp")
    # If write_xml_soup expects a str, use str(tmp_path)
    tdm_parser.write_xml_soup(soup, tmp_path)
    tmp_path.replace(xml_path)  # atomic on POSIX

def _process_one(xml_path: Path) -> Tuple[str, Optional[str]]:
    """
    Worker function.
    Returns (xml_name, error_or_None).
    """
    try:
        soup = tdm_parser.get_xml_soup(xml_path)
        soup = step_tfidf_tags(soup, _EXTRACTOR)
        _atomic_write_xml(soup, xml_path)
        return (xml_path.name, None)
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return (xml_path.name, err)

def tfidf_step_holder(
    corpus_dir: Path,
    log_file_name: str,
    *,
    workers: Optional[int] = None,
    batch_log_size: int = 200,
    batch_log_seconds: float = 10.0,
) -> None:
    """
    Parallel TF-IDF tagging (Python 3.10):
    - pathlib-only file handling
    - atomic writes
    - processes every file (no 'skip unchanged' logic)
    """
    # Build worklist
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger_module.TDMLogger(log_dir=LOGS_PATH, log_file_name = log_file_name, corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    pending = deque(logger_instance.get_file_names())

    xml_paths = [corpus_dir / name for name in pending if (corpus_dir / name).is_file()]
    path_queue = collections.deque(xml_paths)

    processed_buffer: list[str] = []
    last_flush = time.time()

    # Worker count (leave one CPU free)
    if workers is None:
        workers = max(1, (cpu_count() or 2) - 1)

    done = errors = 0

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(TF_IDF_MODEL_PATH,),   # pass Path directly
    ) as ex:
        futures = set()
        while path_queue and len(futures) < workers * 2:
            futures.add(ex.submit(_process_one, path_queue.popleft()))

        for fut in as_completed(futures):
            xml_name, err = fut.result()
            if err is None:
                done += 1
            else:
                errors += 1
                try:
                    if hasattr(logger_instance, "log_error"):
                        logger_instance.log_error(file_name=xml_name, message=err)
                except Exception:
                    pass  # keep going even if error logging fails

                processed_buffer.append(xml_name)

                # Flush by size or time
                now = time.time()
                if len(processed_buffer) >= batch_log_size or (now - last_flush) >= batch_log_seconds:
                    logger_instance.update_log_batch(processed_buffer)
                    processed_buffer.clear()
                    last_flush = now

            futures.difference_update(done_futs)
            while path_queue and len(futures) < workers * 2:
                futures.add(ex.submit(_process_one, path_queue.popleft()))

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)

    print(f"TF-IDF tagging: {done} updated, {errors} errors, workers={workers}")



if __name__ == "__main__":
    corpus_dir = CORPUSES_PATH / 'sample' #'sample' 'LosAngelesTimesDavid' #'LosAngelesTimesDavid' 'Newyork20042023'  TheWashingtonPostDavid  USATodayDavid
    # run is economic step holder
    #is_economic_step_holder(corpus_dir, del_grades=True, prob_threshold=0.5) #TODO # Example usage of the economic step holder
    
    #log_file_name = 'main_step_bert' 
    # = {'negative': 'roberta_title_negative', 'neutral': 'roberta_title_neutral', 'positive': 'roberta_title_positive'}
    #roberta_paragraph_sentiment_label_dict = {'negative': 'roberta_paragraph_negative', 'neutral': 'roberta_paragraph_neutral', 'positive': 'roberta_paragraph_positive'}
    bert_title_label = {'negative': 'bert_title_negative', 'positive': 'bert_title_positive'}
    bert_paragraph_label = {'negative': 'bert_paragraph_negative', 'positive': 'bert_paragraph_positive'}
    # Run the main step with the specified parameters
    #bert_step_holder(corpus_dir, log_file_name, bert_title_label, bert_paragraph_label)
    #roberta_step_holder(corpus_dir=corpus_dir, log_file_name=log_file_name, roberta_title_label=roberta_title_sentiment_label_dict, roberta_paragraph_label=roberta_paragraph_sentiment_label_dict)
    #processed_tags = {'title_negative_prob':'bert_title_negative', 'title_positive_prob':'bert_title_positive', 'paragraph_avg_negative': 'bert_paragraph_negative', 'paragraph_avg_positive': 'bert_paragraph_positive'}
    #csvs_to_xml(corpus_dir, processed_tags, log_file_name)
    log_file_name = 'tf_idf'
    tfidf_step_holder(corpus_dir, log_file_name)
    
    #print all files in dir /home/ec2-user/SageMaker/david/tdm-sentiment/data/corpuses/sample
    my_path = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/data/corpuses/sample')
    my_dir = my_path.glob('*')
    print(my_dir)
