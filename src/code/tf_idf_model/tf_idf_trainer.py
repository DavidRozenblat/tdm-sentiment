import ast
from pathlib import Path
import pandas as pd
from keyword_extractor import TfidfKeywordExtractor
tf_idf_keyword_extractor = TfidfKeywordExtractor()

import joblib
import ast
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def get_doc_str(row, body_text_col, title_col):
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
    # Convert the string representation of a list into an actual list.
    body_text_list = ast.literal_eval(row[body_text_col])
    # Extract the title from the row.
    title_str = row[title_col]
    # Combine the title with the body text list and join with a space.
    doc_str = " ".join([title_str] + body_text_list)
    return doc_str

def get_csv_file_names(path):
    """
    Retrieve sorted CSV file names from a specified folder.
    
    This function assumes that CSV file names contain an underscore followed by an integer 
    at the second position when split by '_'. Sorting is performed based on that integer.
    
    Parameters:
    - result_folder_path: The path to the directory containing CSV files.
    - corpus_name: Corpus identifier (currently not used in the function).
    
    Returns:
    - A list of CSV file names sorted according to the embedded integer.
    """
    # Get all CSV file names in the folder.
    file_list = [f.name for f in path.glob('*.csv')]
    # Sort file names using the integer in the file name.
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[1]))
    return file_list

def get_title_body_str(result_folder_path, corpus_name):
    """
    Process CSV files in the specified folder and return a list of concatenated title-body strings.
    
    For each CSV file, the function reads the data, then processes each row to combine the title 
    and body text (which is stored as a string representation of a list) into a single string.
    
    Parameters:
    - result_folder_path: The path to the directory containing CSV files.
    - corpus_name: Corpus identifier used for filtering or sorting CSV files.
    
    Returns:
    - A list of strings where each string is the concatenation of an article's title and body text.
    """
    title_body_str_lst = []
    csv_file_names_list = get_csv_file_names(result_folder_path / corpus_name)
    
    if not csv_file_names_list:
        print("No CSV files found or modification aborted by the user.")
        return []
    
    for i, file_name in enumerate(csv_file_names_list):
        print(f"Processing chunk {i}: {file_name}")
        file_path = Path(result_folder_path) / file_name
        df = pd.read_csv(file_path)
        
        chunk_results = []
        # Process each row with a progress bar.
        with tqdm(total=len(df), desc=f"Processing chunk {i}", bar_format="{l_bar}{bar:10}{r_bar}") as pbar:
            for _, row in df.iterrows():
                doc_str = get_doc_str(row, 'Text', 'Title')
                chunk_results.append(doc_str)
                pbar.update(1)
        
        title_body_str_lst.extend(chunk_results)
    
    return title_body_str_lst


project_path = Path("c:/Users/pc/Documents/work/bank of israel/financial division/yossi/tdm-sentiment")
data_path = project_path / 'data'
corpus_name = 'LosAngelesTimes_sample20'

lst = get_title_body_str(data_path, corpus_name)

#print(lst)
def train_model(train_lst, tdf_vectorizer):
    """
    Train the vectorizer on the provided training documents and save the model to disk.

    This function takes a list of training documents and an instance of a vectorizer.
    It uses the vectorizer's 'train' method to fit the model to the training data and then
    saves the trained model to a pickle file named 'tfidf_vectorizer.pkl' using the vectorizer's 'save' method.

    Parameters:
    - train_lst (list): A list of training documents (strings) for training the model.
    - tdf_vectorizer: An object that implements a 'train' method for model training and a 'save' method for persisting the model.
    
    Returns: None
    """
    tdf_vectorizer.train(train_lst)
    # Save the trained model to a pickle file.
    tdf_vectorizer.save('tfidf_vectorizer.pkl')

    
# load df of representative sample
texts = [
    "This is a sample document about machine learning and data science.",
    "Another example document discussing natural language processing techniques.",
    "More text data helps the model learn vocabulary and contextual information.",
    "This is the first document. It contains text about machine learning and data science.",
    "The second document focuses on natural language processing and machine learning applications.",
    "Text analysis is a part of data science that includes extracting keywords using TF-IDF.",
    "More documents may discuss different topics such as computer vision, AI, and deep learning."
]
title = "this is the title"



path = data_path / corpus_name
file_list = [f.name for f in path.glob('*.csv')]
print(file_list)

#print(list(path.glob('*.csv')))