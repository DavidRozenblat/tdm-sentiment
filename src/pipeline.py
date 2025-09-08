"""Command-line pipeline for running article processing steps."""
from pathlib import Path
SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
import sys
sys.path.append(str(SRC_PATH))
from config import *  # noqa: F401,F403
from bs4 import BeautifulSoup
from collections import deque
import collections
from typing import Tuple, Optional, List
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import cpu_count
import time
from contextlib import contextmanager
import os
import pandas as pd
from joblib import Parallel, delayed

# -----------------------------
# Parser instance (provided by codebase)
# -----------------------------

tdm_parser = tdm_parser_module.TdmXmlParser()  # Instantiate tdm parser

# =============================
# Helper: temporarily hide CUDA
# =============================
@contextmanager
def masked_cuda_env():
    """Temporarily hide GPUs from child processes (used for TF-IDF pool only)."""
    old = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old

# =============================
# Economic classification (CPU/GPU agnostic)
# =============================

def step_identify_economic(soup: BeautifulSoup,
                           economic_classifier: is_economic_module.EconomicClassifier):
    """For each xml file add probability tag that it's an economic article."""
    text = tdm_parser.get_art_text(soup)
    value = economic_classifier.is_economic_prob(text)
    soup = tdm_parser.modify_tag(soup, tag_name='is_economic', value=value, modify=True)
    return soup


def is_above_threshold(soup: BeautifulSoup, prob_threshold: float) -> bool:
    """Return True iff the stored is_economic prob > threshold."""
    is_econ = tdm_parser.get_tag_value(soup, 'is_economic')
    try:
        return float(is_econ) > prob_threshold
    except (TypeError, ValueError):
        return False


def _is_economic_step(xml_path: Path, classifier, del_grades: bool, prob_threshold: float) -> Tuple[str, bool, Optional[str]]:
    """Helper for economic step; returns (xml_name, is_economic_bool, error_or_None)."""
    try:
        soup = tdm_parser.get_xml_soup(xml_path)
        if del_grades:
             tdm_parser.delete_tag(soup, tag_name='grades')
        soup = step_identify_economic(soup, classifier)
        tdm_parser.write_xml_soup(soup, xml_path)
        return (xml_path.name, is_above_threshold(soup, prob_threshold), None)
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return (xml_path.name, False, err)


def is_economic_step_holder(corpus_dir: Path, del_grades: bool, prob_threshold: float = 0.5) -> None:
    """Add is_economic probability tag; write names above threshold to economic_files.txt."""
    is_economic_classifier = is_economic_module.EconomicClassifier(IS_ECONOMIC_MODEL)  # load classifier
    out_list = FILE_NAMES_PATH / corpus_dir.name / 'economic_files.txt'
    out_list.parent.mkdir(parents=True, exist_ok=True)

    initial_file_list = [xml_file.name for xml_file in corpus_dir.glob('*.xml')]
    logger_instance = logger_module.TDMLogger(
        log_dir=LOGS_PATH,
        log_file_name='is_economic',
        corpus_name=corpus_dir.stem,
        initiate_file_list=initial_file_list,
    )
    pending = deque(logger_instance.get_file_names())
    processed_buffer: List[str] = []

    with out_list.open('a') as f:
        while pending:
            xml_name = pending.popleft()
            xml_path = corpus_dir / xml_name
            try:
                soup = tdm_parser.get_xml_soup(xml_path)
                if del_grades:
                     tdm_parser.delete_tag(soup, tag_name='grades')
                soup = step_identify_economic(soup, is_economic_classifier)
                tdm_parser.write_xml_soup(soup, xml_path)
                if is_above_threshold(soup, prob_threshold):
                    f.write(f"{xml_path.name}\n")
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
            finally:
                processed_buffer.append(xml_name)
                if len(processed_buffer) >= 200:
                    logger_instance.update_log_batch(processed_buffer)
                    processed_buffer.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)

    print(f"Finished processing {len(initial_file_list)} files. Economic articles saved to {out_list}.")



def is_economic_step_holder_parallel(corpus_dir: Path, del_grades: bool, chunk_size: int = 200, prob_threshold: float = 0.5) -> None:
    """Add is_economic probability tag; write names above threshold to economic_files.txt."""
    is_economic_classifier = is_economic_module.EconomicClassifier(IS_ECONOMIC_MODEL)  # load classifier
    out_list = FILE_NAMES_PATH / corpus_dir.name / 'economic_files.txt'
    out_list.parent.mkdir(parents=True, exist_ok=True)

    initial_file_list = [xml_file.name for xml_file in corpus_dir.glob('*.xml')]
    logger_instance = logger_module.TDMLogger(
        log_dir=LOGS_PATH,
        log_file_name='is_economic',
        corpus_name=corpus_dir.stem,
        initiate_file_list=initial_file_list,
    )
    pending_list = logger_instance.get_file_names()
    processed_buffer = []

    with out_list.open('a') as f:
        for i in range(0, len(pending_list), chunk_size):
            processed_buffer = pending_list[i:i+chunk_size]
            
            # Process files in parallel.
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(_is_economic_step)(corpus_dir / name, is_economic_classifier, del_grades, prob_threshold) for name in processed_buffer
            )
            # update economic and log files
            for xml_name, above_threshold, err in results:
                if err:
                    print(f"Error processing {corpus_dir / xml_name}: {err}") # TODO
                elif above_threshold:
                    f.write(f"{xml_name}\n")
    
            logger_instance.update_log_batch(processed_buffer)

    print(f"Finished processing {len(initial_file_list)} files. Economic articles saved to {out_list}.")


# =============================
# Sentiment steps (leave GPU visible; optional device wiring can be added later)
# =============================

def roberta_step_holder(corpus_dir: Path,
                        log_file_name: str,
                        roberta_title_label: dict,
                        roberta_paragraph_label: dict,) -> None:
    roberta_sentiment_analyzer = sentiment_model.TextAnalysis(ROBERTA_MODEL_PATH)  # may use GPU internally

    economic_files = (FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()

    logger_instance = logger_module.TDMLogger(
        log_dir=LOGS_PATH,
        log_file_name=log_file_name,
        corpus_name=corpus_dir.stem,
        initiate_file_list=economic_files,
    )
    pending = deque(logger_instance.get_file_names())
    processed_buffer: List[str] = []

    while pending:
        xml_name = pending.popleft()
        try:
            xml_file = corpus_dir / xml_name
            soup = tdm_parser.get_xml_soup(xml_file)
            soup = step_title_sentiment_prob(soup=soup, sentiment_model=roberta_sentiment_analyzer, label_dict=roberta_title_label)
            soup = step_paragraph_sentiment_prob(soup=soup, sentiment_model=roberta_sentiment_analyzer, label_dict=roberta_paragraph_label)
            tdm_parser.write_xml_soup(soup, xml_file)
        except Exception as e:
            print(f"Error processing {xml_name}: {e}")
        finally:
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
                     bert_paragraph_label: dict,) -> None:
    bert_sentiment_analyzer = sentiment_model.TextAnalysis(BERT_MODEL_PATH)  # may use GPU internally

    economic_files = (FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger_module.TDMLogger(
        log_dir=LOGS_PATH,
        log_file_name=log_file_name,
        corpus_name=corpus_dir.stem,
        initiate_file_list=economic_files,
    )
    pending = deque(logger_instance.get_file_names())
    processed_buffer: List[str] = []

    while pending:
        xml_name = pending.popleft()
        try:
            xml_file = corpus_dir / xml_name
            soup = tdm_parser.get_xml_soup(xml_file)
            soup = step_title_sentiment_prob(soup=soup, sentiment_model=bert_sentiment_analyzer, label_dict=bert_title_label)
            soup = step_paragraph_sentiment_prob(soup=soup, sentiment_model=bert_sentiment_analyzer, label_dict=bert_paragraph_label)
            tdm_parser.write_xml_soup(soup, xml_file)
        except Exception as e:
            print(f"Error processing {xml_name}: {e}")
        finally:
            processed_buffer.append(xml_name)
            if len(processed_buffer) >= 200:
                logger_instance.update_log_batch(processed_buffer)
                processed_buffer.clear()

    if processed_buffer:
        logger_instance.update_log_batch(processed_buffer)

    print(f"Finished processing {len(economic_files)} files in {corpus_dir}.")


def step_title_sentiment_prob(soup: BeautifulSoup,
                              sentiment_model: sentiment_model.TextAnalysis,
                              label_dict: dict) -> BeautifulSoup:
    """Add a sentiment label to each article title."""
    try:
        title = tdm_parser.get_tag_value(soup, 'Title')
        sentiment_dict = sentiment_model.txt_sentiment_dict(title)
        for label in label_dict.keys():
            soup = tdm_parser.modify_tag(soup, label_dict[label], sentiment_dict[label], modify=False)
    except Exception as e:
        print(f"Error processing title sentiment: {e}")
    return soup


def article_average_sentiment_helper(paragraphs, analyzer) -> dict:
    """Compute average positive, neutral, and negative over a list of paragraphs."""
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
                                  label_dict: dict) -> BeautifulSoup:
    """Calculate sentiment probabilities for article paragraphs."""
    try:
        paragraphs_lst = tdm_parser.get_art_text(soup, return_str=False)
        sentiment_dict = article_average_sentiment_helper(paragraphs_lst, sentiment_model)
        for label in label_dict.keys():
            soup = tdm_parser.modify_tag(soup, label_dict[label], sentiment_dict[label], modify=False)
    except Exception as e:
        print(f"Error processing paragraph sentiment: {e}")
    return soup

# ======================================
# TF-IDF (CPU-only pool via masked CUDA)
# ======================================
_EXTRACTOR = None  # initialized per worker

def _init_worker_cpu(tfidf_model_path: Path, stop_words: str = "english") -> None:
    """Load the TF-IDF extractor once per process; ensure GPU is hidden."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # belt-and-suspenders in worker
    global _EXTRACTOR
    _EXTRACTOR = tf_idf_extractor.TfidfKeywordExtractor(
        model_path=tfidf_model_path,
        stop_words=stop_words,
    )


def _atomic_write_xml(soup, xml_path: Path) -> None:
    """Atomic write with pathlib: write to .tmp then replace."""
    tmp_path = xml_path.with_suffix(xml_path.suffix + ".tmp")
    tdm_parser.write_xml_soup(soup, tmp_path)
    tmp_path.replace(xml_path)  # atomic on POSIX


def _process_one(xml_path: Path) -> Tuple[str, Optional[str]]:
    """Worker function. Returns (xml_name, error_or_None)."""
    try:
        soup = tdm_parser.get_xml_soup(xml_path)
        soup = step_tfidf_tags(soup, _EXTRACTOR)
        _atomic_write_xml(soup, xml_path)
        try:
            soup.decompose()
        except Exception:
            pass
        return (xml_path.name, None)
    except Exception as e:
        err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        return (xml_path.name, err)


def step_tfidf_tags(soup: BeautifulSoup,
                    tfidf_extractor: tf_idf_extractor.TfidfKeywordExtractor) -> BeautifulSoup:
    """Append TF-IDF keyword tags to article."""
    text = tdm_parser.get_art_text(soup)
    tf_idf_values = tfidf_extractor.extract_top_keywords(txt_str=text, top_n=200)
    return tdm_parser.modify_tag(soup, 'tf_idf', tf_idf_values, modify=False)


def tfidf_step_holder(
    corpus_dir: Path,
    log_file_name: str,
    *,
    workers: Optional[int] = None,
    batch_log_size: int = 2000,
    batch_log_seconds: float = 40.0,
    window_factor: int = 2,   # how many in-flight tasks per worker
) -> None:
    """
    Parallel TF-IDF tagging (CPU-only workers):
    - windowed scheduling (keeps memory bounded)
    - atomic writes
    - per-process TF-IDF model via _init_worker_cpu
    - robust error logging + periodic flushes
    """

    start_ts = time.time()

    # 1) Build worklist from logger (resume-friendly). Fallback to *.xml if needed.
    economic_files = (FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()

    logger_instance = logger_module.TDMLogger(
        log_dir=LOGS_PATH,
        log_file_name=log_file_name,
        corpus_name=corpus_dir.stem,
        initiate_file_list=economic_files,
    )

    pending_names = collections.deque(logger_instance.get_file_names())
    processed_buffer: List[str] = []
    last_flush = time.time()

    # 2) Worker count (leave one CPU free by default)
    if workers is None:
        workers = max(1, (cpu_count() or 2) - 1)
    inflight_limit = max(1, workers * window_factor)

    done = 0
    errors = 0

    def _flush_if_needed(force: bool = False) -> None:
        nonlocal last_flush
        now = time.time()
        if force or len(processed_buffer) >= batch_log_size or (now - last_flush) >= batch_log_seconds:
            if processed_buffer:
                logger_instance.update_log_batch(processed_buffer)
                print(f"TF-IDF tagging: {len(processed_buffer)} in {(now - last_flush)} seconds, {errors} errors, "
        f"workers={workers}")
                processed_buffer.clear()
            last_flush = now

    # ----------------------
    # CPU-only worker pool
    # ----------------------
    with masked_cuda_env():
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker_cpu,
            initargs=(TF_IDF_MODEL_PATH,),
        ) as ex:
            futures = set()

            # Prefill window
            while pending_names and len(futures) < inflight_limit:
                p = corpus_dir / pending_names.popleft()
                if not p.is_file():
                    continue
                f = ex.submit(_process_one, p)
                setattr(f, "_xml_name", p.name)
                futures.add(f)

            # Main drain/refill loop
            while futures:
                done_set, not_done_set = wait(futures, return_when=FIRST_COMPLETED)

                for fut in done_set:
                    try:
                        xml_name, err = fut.result()
                    except Exception as e:
                        xml_name = getattr(fut, "_xml_name", "<unknown>")
                        err = f"worker crashed: {e}"

                    if err is None:
                        done += 1
                    else:
                        errors += 1
                        try:
                            if hasattr(logger_instance, "log_error"):
                                logger_instance.log_error(file_name=xml_name, message=err)
                        except Exception:
                            pass  # never let logging halt the pipeline

                    processed_buffer.append(xml_name)
                    _flush_if_needed(force=False)

                futures = not_done_set

                # Refill
                while pending_names and len(futures) < inflight_limit:
                    p = corpus_dir / pending_names.popleft()
                    if not p.is_file():
                        continue
                    f = ex.submit(_process_one, p)
                    setattr(f, "_xml_name", p.name)
                    futures.add(f)

    # Final flush
    _flush_if_needed(force=True)

    dt = time.time() - start_ts
    total = done + errors
    rate = (total / dt) if dt > 0 else 0.0
    print(
        f"All {corpus_dir.stem} files done. TF-IDF tagging: {done} ok, {errors} errors, "
        f"workers={workers}, elapsed={dt:.1f}s, rate={rate:.2f} files/s"
    )

# =============================
# CSV -> XML tag updater (unchanged scaffolding)
# =============================

def csvs_to_xml(corpus_dir: Path, processed_tags: dict, log_file_name: str) -> None:
    """Update new tags to XML files from a folder of CSV result files.
    processed_tags is a dict with column names as keys and tag names as values.
    """
    csv_dir = RESULTS_PATH / corpus_dir.name
    csv_file_paths = list(csv_dir.glob('*.csv'))

    try:
        economic_files = (FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    except FileNotFoundError:
        economic_files = [p.name for p in corpus_dir.glob('*.xml')]

    logger_instance = logger_module.TDMLogger(
        log_dir=LOGS_PATH,
        log_file_name=log_file_name,
        corpus_name=corpus_dir.stem,
        initiate_file_list=economic_files,
    )
    xml_file_names = list(logger_instance.get_file_names())

    for csv_path in csv_file_paths:
        try:
            xml_processed = file_process.csv_to_xml(csv_path=csv_path,
                                                    corpus_dir=corpus_dir,
                                                    processed_tags=processed_tags,
                                                    xml_file_names=xml_file_names)
            logger_instance.update_log_batch(xml_processed)
        except Exception as e:
            print(f"Error processing {csv_path.stem}: {e}")

    print(f"Finished processing all {corpus_dir.stem} csv files.")


# =============================
# XML -> CSV 
# =============================

def xmls_to_csv(corpus_dir: Path, log_file_name: str, processed_tags: list = [], chunk_size: int = 2000) -> None:
    """Convert XML files in a folder to CSV format."""
    economic_files = (FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()

    logger_instance = logger_module.TDMLogger(
        log_dir=LOGS_PATH,
        log_file_name=log_file_name,
        corpus_name=corpus_dir.stem,
        initiate_file_list=economic_files,
    )
    economic_files_list = logger_instance.get_file_names()
    
    # get highest number of existing chunk files
    d = RESULTS_PATH / corpus_dir.stem
    max_num = max(
        (int(p.stem.rsplit("_", 2)[1]) for p in d.glob("chunk_*_data.csv")),
        default=-1,
    )
    print(f'max existing chunk number: {max_num}')
    for i in range(0, len(economic_files_list), chunk_size):
        processed_buffer = economic_files_list[i:i+chunk_size]
        data = file_process.xml_to_csv(corpus_dir, processed_buffer, processed_tags)

        output_file = RESULTS_PATH / corpus_dir.stem / f'chunk_{(i // chunk_size) + 1 + max_num}_data.csv'
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        print(f"Saving chunk {(i // chunk_size) + 1 + max_num} to {output_file}") # TODO logging
        data.to_csv(output_file, index=False)  # save df to csv

        logger_instance.update_log_batch(processed_buffer)
    print(f"Finished processing all {corpus_dir.stem} xml files.")

# =============================
# __main__ (example usage)
# =============================
if __name__ == "__main__":
    import logging
    corpus_dir = CORPUSES_PATH / 'sample'  # 'sample' 'LosAngelesTimesDavid', 'USATodayDavid', 'Newyork20042023
    log_file_name = 'xml_to_csv'
    
    # set logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    handler = logging.FileHandler(filename=LOGS_PATH / f'{log_file_name}.log') 
    formatter =logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    #--- run XML to CSV

    processed_tags = ['tf_idf']

    #xmls_to_csv(corpus_dir, log_file_name, processed_tags)

    # Economic step example
    # is_economic_step_holder(corpus_dir, del_grades=True, prob_threshold=0.5)
    is_economic_step_holder_parallel(corpus_dir, del_grades=True, prob_threshold=0.1)
    # Sentiment examples
    # roberta_title = {'negative': 'roberta_title_negative', 'neutral': 'roberta_title_neutral', 'positive': 'roberta_title_positive'}
    # roberta_para  = {'negative': 'roberta_paragraph_negative', 'neutral': 'roberta_paragraph_neutral', 'positive': 'roberta_paragraph_positive'}
    # roberta_step_holder(corpus_dir, 'roberta', roberta_title, roberta_para)

    # bert_title = {'negative': 'bert_title_negative', 'positive': 'bert_title_positive'}
    # bert_para  = {'negative': 'bert_paragraph_negative', 'positive': 'bert_paragraph_positive'}
    # bert_step_holder(corpus_dir, 'bert', bert_title, bert_para)

    # CSV -> XML example
    # processed_tags = {
    #     'title_negative_prob': 'bert_title_negative',
    #     'title_positive_prob': 'bert_title_positive',
    #     'paragraph_avg_negative': 'bert_paragraph_negative',
    #     'paragraph_avg_positive': 'bert_paragraph_positive',
    # }
    # csvs_to_xml(corpus_dir, processed_tags, 'csv_to_xml')

    # TF-IDF (CPU-only pool)
    #tfidf_step_holder(corpus_dir, log_file_name)

    # Example: list files (avoid printing generator)
    # my_path = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/data/corpuses/sample')
    # print(list(my_path.glob('*')))

    
    #processed_tags = ['is_economic', 'tf_idf',
                      #'bert_title_negative', 'bert_title_positive', 'bert_paragraph_negative', 'bert_paragraph_positive',
                      #'roberta_title_negative', 'roberta_title_neutral', 'roberta_title_positive', 'roberta_paragraph_negative', 'roberta_paragraph_neutral', 'roberta_paragraph_positive',
                      #]


