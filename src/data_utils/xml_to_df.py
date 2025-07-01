# set environment
from pathlib import Path
SRC_PATH = Path('/home/ec2-user/SageMaker/david/tdm-sentiment/src/')
import sys
sys.path.append(str(SRC_PATH))
from config import *

import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# Initialize variables
parser = tdm_parser.TdmXmlParser()

TDM_PROPERTY_TAGS = [
    'GOID', 'SortTitle', 'NumericDate', 'mstar', 'DocSection', 
    'GenSubjTerm', 'StartPage', 'WordCount', 'Title', 'Text',
    'CompanyName', 'is_economic', 'bert_sentiment' 
    ] 
PROPERTY_NAMES = [
    'goid', 'publisher', 'date', 'article_type', 'section', 
    'tdm_topic_tags', 'page', 'word_count', 'title', 'paragrph_text',
    'company_name', 'is_economic', 'bert_sentiment',
    ]


def read_file_names_in_chunks(input_file, chunk_size):
    # Open the file and yield file names in chunks.
    with open(f'{input_file}.txt', 'r') as f:
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

def xml_to_dict(file_path):
    """
    Parse the XML file and convert its contents into a dictionary,
    including custom tags like 'is_economic' and 'bert_sentiment'.
    """
    soup = parser.get_xml_soup(file_path)
    content_dict = parser.soup_to_dict(
        soup, tdm_property_tags=TDM_PROPERTY_TAGS, property_names=PROPERTY_NAMES
    )
    return content_dict

def xml_to_df(file_names_path, results_path, chunk_size=2000):
    for i, file_chunk in enumerate(read_file_names_in_chunks(file_names_path, chunk_size)):
        # Construct full paths for each file name.
        chunk_paths = [CORPUSES_PATH / file_name for file_name in file_chunk]
        
        # Process files in parallel with a progress bar.
        with tqdm(total=len(chunk_paths), desc=f"Processing chunk {i}") as pbar: 
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(xml_to_dict)(path) for path in chunk_paths
            )
            pbar.update(len(chunk_paths))

        if results:
            data = pd.DataFrame(results, columns=PROPERTY_NAMES)
            output_file = results_path / f'chunk_{i}_data.csv'
            data.to_csv(output_file, index=False)
            del data

            
def load_df_from_xml(file_names_path, chunk_size=2000):
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
        chunk_paths = [CORPUSES_PATH / file_name for file_name in file_chunk]

        # Process files in parallel with a progress bar.
        with tqdm(total=len(chunk_paths), desc=f"Processing chunk {i}") as pbar:
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(xml_to_dict)(path) for path in chunk_paths
            )

        # Append results if available.
        if results:
            results_lst.extend(results)
    
    # Return a DataFrame or an empty DataFrame if no results.
    if results_lst:
        return pd.DataFrame(results_lst, columns=PROPERTY_NAMES)
    else:
        return pd.DataFrame(columns=PROPERTY_NAMES)

            
            
