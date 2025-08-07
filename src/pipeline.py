


"""Command-line pipeline for running article processing steps.
Each function corresponds to a stage in the end-to-end workflow. 
"""
from pathlib import Path
from config import *
from bs4 import BeautifulSoup
from collections import deque
# Instantiate logger for pipeline steps

tdm_parser = tdm_parser_module.TdmXmlParser()




def step_identify_economic(soup: BeautifulSoup, 
                           economic_classifier: is_economic_module.EconomicClassifier, 
                           del_grades: bool = False): 
    """for each xml file on corpus add probability tag that it's economic article."""
    if del_grades:
        tdm_parser.delete_tag(soup, tag_name='grades')  

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
    initial_file_list = [xml_file.name for xml_file in corpus_dir.glob('*.xml')] #TODO change
    logger_instance = logger.Logger(log_dir=LOGS_PATH, log_file_name = 'is_economic', corpus_name = corpus_dir.stem, initiate_file_list = initial_file_list)
    pending = deque(logger_instance.get_file_names())
    
    # Loop through each XML file in the corpus directory
    with open(file_path, 'a') as f:
        while pending:
            xml_name = pending.popleft()
            try:            
                xml_path = corpus_dir / xml_name
                goid = xml_path.stem
                soup = tdm_parser.get_xml_soup(xml_path)
                soup = step_identify_economic(soup, is_economic_classifier, del_grades=del_grades)
                # rewrite to file
                tdm_parser.write_xml_soup(soup, xml_path)
                # Check if the article is economic and write to file if it is
                if is_above_threshold(soup, prob_threshold): 
                    f.write(f"{xml_path.name}\n")
            except Exception as e:
                print(f"Error processing {xml_path}: {e}")
            finally:
                # 4) update log regardless of success/failure
                logger_instance.update_log_file(xml_name)

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


def article_average_sentiment_helper(paragraphs, analyzer): #todo
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
    tfidf_extractor = tf_idf_extractor.TfidfKeywordExtractor(TF_IDF_MODEL_PATH)  # load the TF-IDF extractor model
    roberta_sentiment_analyzer = sentiment_model.TextAnalysis(ROBERTA_MODEL_PATH)  # load the sentiment model
    bert_sentiment_analyzer = sentiment_model.TextAnalysis(BERT_MODEL_PATH)  # load the sentiment model
    
    # get the list of economic files and initialize logger
    economic_files = Path(FILE_NAMES_PATH / corpus_dir.stem / "economic_files.txt").read_text().splitlines()
    logger_instance = logger.Logger(log_dir=LOGS_PATH, log_file_name = log_file_name, corpus_name = corpus_dir.stem, initiate_file_list = economic_files)
    pending = deque(logger_instance.get_file_names())

    while pending:
        xml_name = pending.popleft()
        try:
            xml_file = corpus_dir / xml_name
            soup = tdm_parser.get_xml_soup(xml_file)
            soup = step_tfidf_tags(soup=soup, tfidf_extractor=tfidf_extractor)  # append TF-IDF tags
            soup = step_title_sentiment_prob(soup=soup, sentiment_model=roberta_sentiment_analyzer, label_dict=roberta_title_sentiment_label_dict)  # add title sentiment
            soup = step_paragraph_sentiment_prob(soup=soup, sentiment_model=roberta_sentiment_analyzer, label_dict=roberta_paragraph_sentiment_label_dict)  # add paragraph sentiment
            soup = step_title_sentiment_prob(soup=soup, sentiment_model=bert_sentiment_analyzer, label_dict=bert_title_sentiment_label_dict)  # add title sentiment
            soup = step_paragraph_sentiment_prob(soup=soup, sentiment_model=bert_sentiment_analyzer, label_dict=bert_paragraph_sentiment_label_dict)  # add paragraph sentiment
            # rewrite to file
            tdm_parser.write_xml_soup(soup, xml_file)
        except Exception as e:
            print(f"Error processing {xml_name}: {e}")
        finally:
            # 4) update log regardless of success/failure
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
    pass



if __name__ == '__main__':
    #main()
    corpus_dir = CORPUSES_PATH / 'sample'#'LosAngelesTimesDavid' #'LosAngelesTimesDavid' # 'Newyork20042023'  TheWashingtonPostDavid  USATodayDavid
    # run is economic step holder
    #is_economic_step_holder(corpus_dir, del_grades=True, prob_threshold=0.2) #TODO # Example usage of the economic step holder
    
    log_file_name = 'main_step_tf_idf_roberta_bert_sentiment' 
    roberta_title_sentiment_label_dict = {'negative': 'roberta_title_negative', 'neutral': 'roberta_title_neutral', 'positive': 'roberta_title_positive'}
    roberta_paragraph_sentiment_label_dict = {'negative': 'roberta_paragraph_negative', 'neutral': 'roberta_paragraph_neutral', 'positive': 'roberta_paragraph_positive'}
    bert_title_sentiment_label_dict = {'negative': 'bert_title_negative', 'neutral': 'bert_title_neutral', 'positive': 'bert_title_positive'}
    bert_paragraph_sentiment_label_dict = {'negative': 'bert_paragraph_negative', 'neutral': 'bert_paragraph_neutral', 'positive': 'bert_paragraph_positive'}
    # Run the main step with the specified parameters
    
    main_step_holder(corpus_dir=corpus_dir,
                    log_file_name=log_file_name, 
                    roberta_title_sentiment_label_dict=roberta_title_sentiment_label_dict, 
                    roberta_paragraph_sentiment_label_dict=roberta_paragraph_sentiment_label_dict,
                    bert_title_sentiment_label_dict=bert_title_sentiment_label_dict,
                    bert_paragraph_sentiment_label_dict=bert_paragraph_sentiment_label_dict)
    
    
    