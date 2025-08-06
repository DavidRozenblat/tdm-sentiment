# set environment
from pathlib import Path
SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
import sys
sys.path.append(str(SRC_PATH))
from config import *

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import random
# Initialize variables
parser = tdm_parser_module.TdmXmlParser()

TDM_PROPERTY_TAGS = [
    'GOID', 'SortTitle', 'NumericDate', 'mstar', 'DocSection', 
    'GenSubjTerm', 'StartPage', 'WordCount', 'Title', 'Text',
    'CompanyName'
    ] 
PROPERTY_NAMES = [
    'goid', 'publisher', 'date', 'article_type', 'section', 
    'tdm_topic_tags', 'page', 'word_count', 'title', 'paragrph_text',
    'company_name', 
    ]



def get_file_paths_sample(corpuses_dir: Path, output_path: Path, sample_size: int):
    """
    Recursively glob all XML files under corpuses_dir, take a random sample
    of up to sample_size paths, and write them (one per line) to output_path.
    """
    # 1. Ensure the parent directory for the output file exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Collect all XML files (recursive)
    xml_files = list(corpuses_dir.rglob('*.xml'))
    if not xml_files:
        raise ValueError(f"No XML files found in {corpuses_dir}")

    # 3. Sample up to sample_size (no error even if sample_size > total files)
    count = min(sample_size, len(xml_files))
    sampled_files = random.sample(xml_files, count)

    # 4. Write out
    with output_path.open('w', encoding='utf-8') as f:
        for file_path in sampled_files:
            f.write(f"{file_path}\n")
            

def read_file_names_in_chunks(input_file, chunk_size):
    # Open the file and yield file names in chunks.
    with open(f'{input_file}', 'r') as f:
        chunk = []
        for i, line in enumerate(f, 1):
            file_name = line.strip()
            if file_name:
                chunk.append(file_name)
            if i % chunk_size == 0:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def xml_to_dict(file_path: Path, processed_tags: list):
    """
    Parse the XML file and convert its contents into a dictionary,
    including custom tags like 'is_economic' and 'bert_sentiment'.
    """
    soup = parser.get_xml_soup(file_path)
    content_dict = parser.soup_to_dict(
        soup, tdm_property_tags=TDM_PROPERTY_TAGS + processed_tags, property_names=PROPERTY_NAMES + processed_tags
    )
    return content_dict


def xml_to_csv(corpus_dir: Path, file_names_path: Path, processed_tags: list = [], chunk_size: int = 2000):
    """
    Load XML files in chunks and return a DataFrame.
    """
    for i, file_chunk in enumerate(read_file_names_in_chunks(file_names_path, chunk_size)):
        # Construct full paths for each file name.
        chunk_paths = [corpus_dir / file_name for file_name in file_chunk]

        # Process files in parallel with a progress bar.
        with tqdm(total=len(chunk_paths), desc=f"Processing chunk {i}") as pbar: 
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(xml_to_dict)(path, processed_tags) for path in chunk_paths
            )
            pbar.update(len(chunk_paths))

        if results:
            data = pd.DataFrame(results, columns=PROPERTY_NAMES + processed_tags)
            output_file = RESULTS_PATH / corpus_dir.stem / f'chunk_{i}_data.csv'
            output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            print(f"Saving chunk {i} to {output_file}")
            data.to_csv(output_file, index=False)  # save df to csv
            del data


def load_df_from_xml(file_names_path: Path, chunk_size: int = 2000, processed_tags: list = []):
    """
    Load XML files in chunks and return a DataFrame.
    
    Parameters:
    - file_names_path: Path to the file containing the list of XML file names.
    - chunk_size: The number of file names to process in each chunk.

    Returns:
    - A pandas DataFrame constructed from the XML file data, or an empty DataFrame if no data is found.
    """    
    results_lst = []
    for i, file_chunk in enumerate(read_file_names_in_chunks(file_names_path, chunk_size)):
        # Construct full paths for each file name.
        chunk_paths = [file_name for file_name in file_chunk]

        # Process files in parallel with a progress bar.
        with tqdm(total=len(chunk_paths), desc=f"Processing chunk {i}") as pbar:
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(xml_to_dict)(path, processed_tags) for path in chunk_paths
            )

        # Append results if available.
        if results:
            results_lst.extend(results)
    
    # Return a DataFrame or an empty DataFrame if no results.
    if results_lst:
        return pd.DataFrame(results_lst, columns=PROPERTY_NAMES + processed_tags)
    else:
        return pd.DataFrame(columns=PROPERTY_NAMES + processed_tags)


def update_csv_with_new_tags(csv_path: Path, corpus_dir: Path, processed_tags: list):
    """
    Update the CSV file with new tags from the XML files in the corpus.
    
    Parameters:
    - csv_path: Path to the existing CSV file.
    - corpus_path: Path to the directory containing XML files.
    - processed_tags: List of tags to be added to the DataFrame.
    
    Returns:
    - None
    """
    df = pd.read_csv(csv_path)
    
    for tag in processed_tags:
        if tag not in df.columns:
            df[tag] = None  # Initialize new columns with None
    
    # match XML using GOID tag and goid column
    goid_list = df['goid'].tolist()# make it a list of strings
    goid_list = [str(goid) for goid in goid_list]
    xml_file_path_list = [file for file in corpus_dir.glob('*.xml') if file.stem in goid_list]
    for xml_file in xml_file_path_list:
        soup = parser.get_xml_soup(xml_file)
        goid = xml_file.stem
        if soup:
            content_dict = parser.soup_to_dict(soup, tdm_property_tags=processed_tags, property_names=processed_tags)
            if goid in df['goid'].values:
                for tag in processed_tags:
                    df.loc[df['goid'] == goid, tag] = content_dict.get(tag)
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    # Example usage
    corpus_name = 'USATodayDavid'  #'LosAngelesTimesDavid'  
    csv_path = RESULTS_PATH / corpus_name / 'chunk_0_data.csv'
    corpus_dir = CORPUSES_PATH / corpus_name
    processed_tags = ['tf_idf','roberta_title_negative', 'roberta_title_neutral', 'roberta_title_positive']
    
    # Update the CSV with new tags
    #update_csv_with_new_tags(csv_path, corpus_dir, processed_tags)
    file_names_path = FILE_NAMES_PATH / corpus_dir.stem / 'economic_files'
    #xml_to_csv(corpus_dir=corpus_dir, file_names_path=file_names_path, processed_tags=processed_tags, chunk_size=2000)


    #'is_economic', 'tf_idf', 'bert_title_negative', 'bert_title_neutral', 'bert_title_positive',
    # 'roberta_title_negative', 'roberta_title_neutral', 'roberta_title_positive'

    # Define file paths
    corpuses_dir = CORPUSES_PATH  #'all_dataset_file_names.txt' LosAngelesTimesDavid all_dataset_file_names.txt # Path to the input text file
    output_path = FILE_NAMES_PATH / 'all_files.txt'  # Path to the output text file 
    
    # Example usage
    #get_file_paths_sample(corpuses_dir, output_path, sample_size=1000)
    
    file_names_path = FILE_NAMES_PATH / 'all_files.txt'  # Path to the file containing the list of XML file name
    df = load_df_from_xml(file_names_path)
    print(df.head())