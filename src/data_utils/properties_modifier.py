from pathlib import Path
import sys

# Define the source path and add it to sys.path.
SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src')

sys.path.append(str(SRC_PATH))

# Import configuration and other required modules.
from config import *  # Make sure config defines: TextAnalysis, SENTIMENT_MODEL_PATH, LOGS_PATH, etc.
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm 
import ast
from concurrent.futures import ProcessPoolExecutor
from bs4 import BeautifulSoup

# Create a Logger instance
tdm_parser = tdm_parser_module.TdmXmlParser()

publisher_corpus_dic = {
    'New York Times':'Newyork20042023', 'Los Angeles Times':'LosAngelesTimesDavid', 
    'Washington Post The':'TheWashingtonPostDavid','Chicago Tribune': 'ChicagoTribune', 
    'USA Today (Online)':'USATodayDavid', 'USA Today':'USATodayDavid'
}

# --- Processing Functions for title sentiment---
def title_sentiment(text, sentiment_analyzer):
    """
    Analyze the sentiment of a single title text.
    """
    # Note: The function expects a list of texts; we wrap text in a list.
    try:
        return sentiment_analyzer.analyze_article_sentiment([text], method='bert')
    except Exception as e:
        print(f'error: {e}, input was:{text}')
        return None


def modify_csv_title_sentiment(result_folder_path, corpus_name, sentiment_analyzer):
    """
    Process CSV files by adding a 'title_sentiment' column using sentiment analysis.
    """
    log_file_names_list = tdm_logger.get_logger_file_names('modify_sentiment_title', result_folder_path, corpus_name)
    if log_file_names_list is None:
        print("Modification aborted by the user.")
        return

    # Build a list of unprocessed files.
    unprocessed_files = list(log_file_names_list)  # copy

    for i, file_name in enumerate(log_file_names_list):
        print(f"Processing chunk {i}: {file_name}")
        file_path = Path(result_folder_path) / file_name

        # Read the CSV file.
        df = pd.read_csv(file_path)

        # Extract titles from the DataFrame.
        #texts = df['title'].tolist()

        # analyze title sentiment
        with tqdm(total=len(df), desc=f"Processing chunk {i}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            #results = Parallel(n_jobs=-1, backend='threading')(
                #delayed(title_sentiment)(text, sentiment_analyzer) for text in texts
            #)
            df['title_sentiment'] = df['title'].apply(lambda text: title_sentiment(text, sentiment_analyzer))
            pbar.update(len(df))  # Update progress bar after completion.

        # Assign the sentiment results back to the DataFrame.
        #df['title_sentiment'] = results

        # Save the modified DataFrame back to CSV.
        df.to_csv(file_path, index=False)

        # Remove the processed file from the unprocessed_files list.
        if file_name in unprocessed_files:
            unprocessed_files.remove(file_name)

        # Update the log file with the remaining files.
        tdm_logger.update_log_file(result_folder_path, 'modify_sentiment_title', unprocessed_files, corpus_name)

    print("All files processed successfully!")


# --- Processing Functions for title sentiment prob---
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


def modify_csv_title_sentiment_prob(result_folder_path, corpus_name, sentiment_analyzer):
    """
    Process CSV files by adding a 'title_sentiment' column using sentiment analysis.
    """
    log_file_names_list = tdm_logger.get_logger_file_names('modify_sentiment_title_prob', result_folder_path, corpus_name)
    if log_file_names_list is None:
        print("Modification aborted by the user.")
        return

    # Build a list of unprocessed files.
    unprocessed_files = list(log_file_names_list)  # copy

    for i, file_name in enumerate(log_file_names_list):
        print(f"Processing chunk {i}: {file_name}")
        file_path = Path(result_folder_path) / file_name

        # Read the CSV file.
        df = pd.read_csv(file_path)

        # Analyze title sentiment for each row.
        # Use .apply to get a Series of dictionaries.
        with tqdm(total=len(df), desc=f"Processing chunk {i}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            sentiment_series = df['title'].apply(lambda text: title_sentiment_probs(text, sentiment_analyzer))
            # Convert the Series of dictionaries into a DataFrame of columns.
            sentiment_df = sentiment_series.apply(pd.Series)
            # Assign each sentiment probability to a new column. Use .get() in case some values are missing.
            df['title_positive_prob'] = sentiment_df.get('positive', 0)
            df['title_neutral_prob'] = sentiment_df.get('neutral', 0)
            df['title_negative_prob'] = sentiment_df.get('negative', 0)
            pbar.update(len(df))  # update the progress bar


        # Save the modified DataFrame back to CSV.
        df.to_csv(file_path, index=False)

        # Remove the processed file from the unprocessed_files list.
        if file_name in unprocessed_files:
            unprocessed_files.remove(file_name)

        # Update the log file with the remaining files.
        tdm_logger.update_log_file(result_folder_path, 'modify_sentiment_title_probs', unprocessed_files, corpus_name)

    print("All files processed successfully!")
    

# --- Processing Functions for paragraph-level sentiment probabilities ---
def text_sentiment_probs(text, sentiment_analyzer):
    """
    Get the sentiment probabilities of a single text (e.g., title, paragraph sentence).
    Returns a dict with keys 'positive', 'neutral', 'negative'.
    """
    if len(text) > 512:
        print(f'text Truncate, len text is{len(text)}')
        text = text[:511]
    try:
        return sentiment_analyzer.get_sentiment_dict(text)
    except Exception as e:
        print(f"Error during sentiment analysis: {e} | input: {text}")
        return None


def modify_csv_paragraph_sentiment_prob(result_folder_path, corpus_name, sentiment_analyzer):
    """
    Process CSV files by:
      1. Parsing the 'paragrph_text' column (stringified list) back into a Python list.
      2. Computing sentiment probabilities for each paragraph entry.
      3. Averaging the positive, neutral, and negative probabilities across all paragraphs.
      4. Adding three new columns: 'paragraph_avg_positive',
         'paragraph_avg_neutral', 'paragraph_avg_negative'.
    """
    # Retrieve files to process
    log_file_names_list = tdm_logger.get_logger_file_names(
        'modify_sentiment_paragraph_prob',
        result_folder_path,
        corpus_name
    )
    if log_file_names_list is None:
        print("Modification aborted by the user.")
        return

    unprocessed_files = list(log_file_names_list)

    for i, file_name in enumerate(log_file_names_list):
        print(f"Processing chunk {i}: {file_name}")
        file_path = Path(result_folder_path) / file_name

        # Read CSV into DataFrame
        df = pd.read_csv(file_path)

        # Prepare lists to collect averaged scores
        avg_positives = []
        avg_neutrals  = []
        avg_negatives = []

        # Iterate rows
        with tqdm(total=len(df), desc=f"Chunk {i}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            for text_list_str in df['paragrph_text']:
                # Safely parse the stringified list
                try:
                    paragraphs = ast.literal_eval(text_list_str)
                    if not isinstance(paragraphs, list):
                        raise ValueError("Parsed object is not a list.")
                except Exception:
                    paragraphs = []

                # sentiment accumulators
                pos_vals = []
                neu_vals = []
                neg_vals = []

                # Analyze each paragraph entry
                for para in paragraphs:
                    probs = text_sentiment_probs(para, sentiment_analyzer)
                    if probs:
                        pos_vals.append(probs.get('positive', 0))
                        neu_vals.append(probs.get('neutral',  0))
                        neg_vals.append(probs.get('negative', 0))

                # Compute simple means (or zero if no paragraphs)
                if pos_vals:
                    avg_positives.append(sum(pos_vals) / len(pos_vals))
                    avg_neutrals.append(sum(neu_vals) / len(neu_vals))
                    avg_negatives.append(sum(neg_vals) / len(neg_vals))
                else:
                    avg_positives.append(0)
                    avg_neutrals.append(0)
                    avg_negatives.append(0)

                pbar.update(1)

        # Assign new columns to DataFrame
        df['paragraph_avg_positive'] = avg_positives
        df['paragraph_avg_neutral']  = avg_neutrals
        df['paragraph_avg_negative'] = avg_negatives

        # Save modified DataFrame
        df.to_csv(file_path, index=False)

        # Update unprocessed_files and log
        if file_name in unprocessed_files:
            unprocessed_files.remove(file_name)
        tdm_logger.update_log_file(
            result_folder_path,
            'modify_sentiment_paragraph_prob',
            unprocessed_files,
            corpus_name
        )

    print("All paragraph sentiment probability files processed successfully!")



# --- Processing Functions for TfIdf  ---
def get_doc_str(row, body_text_col='paragrph_text', title_col='title'):
    """
    Combine an article's title and body text into a single string.
    
    The body text column is expected to store a string representation of a list.
    This function converts that string to a list, prepends the title, and then joins all parts into one string.
    
    Parameters:
    - row: A pandas Series representing a DataFrame row.
    - body_text_col: Name of the column containing the body text.
    - title_col: Name of the column containing the title.
    
    Returns:
    - A string combining the title and the evaluated body text.
    """
    try:
        # Convert the string representation of a list into an actual list.
        #print(row)
        body_text_list = ast.literal_eval(row[body_text_col]) 
        # Extract the title from the row.
        title_str = row[title_col]
        # Combine the title with the body text list and join with a space.
        doc_str = " ".join([title_str] + body_text_list)
        return doc_str
    except Exception as e:
        print(f'exception: {e}')
        return None


def tf_idf_tags(text, tf_idf_extractor, top_n):
    """
    get tf_idf n top words of a single row (title + text).
    """
    try:
        top_words = tf_idf_extractor.extract_keywords_from_documents([text], top_n)[0]
        return top_words
    except Exception as e:
        print(f'error: {e}, input was:{text}')
        return None

    
def modify_csv_tf_idf(result_folder_path, corpus_name, tf_idf_extractor, top_n):
    """
    Process CSV files by adding a `tf_idf_tags` column.
    """
    log_file_names_list = tdm_logger.get_logger_file_names('tf_idf_tags', result_folder_path, corpus_name)
    if log_file_names_list is None:
        print("Modification aborted by the user.")
        return
    
    # Build a list of unprocessed files.
    unprocessed_files = list(log_file_names_list)  # copy

    for i, file_name in enumerate(log_file_names_list):
        texts = []
        print(f"Processing chunk {i}: {file_name} {corpus_name}")
        file_path = Path(result_folder_path) / file_name
        df = pd.read_csv(file_path) # Read the CSV file.
        # Create a list of rows to process.
        # Each row is passed to tf_idf_tags along with the tf_idf_extractor.
        texts = df.apply(get_doc_str, axis=1)
        
        with tqdm(total=len(df), desc=f"Processing chunk {i}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            # Use ProcessPoolExecutor to parallelize the row processing.
            with ProcessPoolExecutor() as executor:
                results = list(executor.map( # Map the tf_idf_tags function to each row.
                    tf_idf_tags,
                    texts,
                    [tf_idf_extractor]*len(texts),  # pass the same extractor to each row
                    [top_n]*len(texts)
                ))
            pbar.update(len(df)) # Update progress bar after completion
        
        # Add the computed TF-IDF tags to the DataFrame.
        df['tf_idf_tags'] = results
        # Save the modified DataFrame back to CSV.
        df.to_csv(file_path, index=False)
         
        # Remove the processed file from the unprocessed_files list.
        if file_name in unprocessed_files:
            unprocessed_files.remove(file_name)
        # Update the log file with the remaining files.
        tdm_logger.update_log_file(result_folder_path, 'tf_idf_tags', unprocessed_files, corpus_name)
    print("All files processed successfully!")
    
    
    
#---- add a TDM tag ------
def tag_to_add(poblisher, tag_name, goid):
    """
    read xml tag 
    """
    poblisher = publisher_corpus_dic[poblisher.strip()]
    file_path = CORPUSES_PATH / poblisher / f'{goid}.xml'
    soup = tdm_parser.get_xml_soup(file_path)
    # Attempt to find the tag in the soup
    tag = soup.find(tag_name)
    # If found, get the text; if not, store None
    tag = tag.get_text(strip=True) if tag else None
    return tag

def add_a_tdm_tag_to_csv(result_folder_path, corpus_name, tag_name):
    """
    Process CSV files by adding a `tag from tdm` column.
    """
    log_file_names_list = tdm_logger.get_logger_file_names(f'adding_{tag_name}', result_folder_path, corpus_name)
    if log_file_names_list is None:
        print("Modification aborted by the user.")
        return
    
    # Build a list of unprocessed files.
    unprocessed_files = list(log_file_names_list)  # copy

    for i, file_name in enumerate(log_file_names_list):
        texts = []
        print(f"Processing chunk {i}: {file_name} {corpus_name}")
        file_path = Path(result_folder_path) / file_name
        df = pd.read_csv(file_path) # Read the CSV file.
        # Each row is passed to tag_to_add 
        goids = df['goid']
        poblishers = df['publisher'] 

        # Create a list of rows to process
        with tqdm(total=len(df), desc=f"Processing chunk {i}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            # Use ProcessPoolExecutor to parallelize the row processing.
            with ProcessPoolExecutor() as executor:
                results = list(executor.map( # Map the tag_to_add function to each row.
                    tag_to_add,
                    poblishers,
                    [tag_name]*len(goids),
                    goids
                ))
            pbar.update(len(df)) # Update progress bar after completion
        
        # Add the computed tags to the DataFrame.
        df[f'{tag_name.lower()}'] = results
        # Save the modified DataFrame back to CSV.
        df.to_csv(file_path, index=False)
         
        # Remove the processed file from the unprocessed_files list.
        if file_name in unprocessed_files:
            unprocessed_files.remove(file_name)
        # Update the log file with the remaining files.
        tdm_logger.update_log_file(result_folder_path, f'adding_{tag_name}', unprocessed_files, corpus_name)
    print("All files processed successfully!")









