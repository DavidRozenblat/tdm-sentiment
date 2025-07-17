# TDM Sentiment Analysis Project

## Project Overview

This project implements a comprehensive pipeline for analyzing sentiment and topics in news articles from major US newspapers, with a focus on financial and economic content. It leverages TDM Studio (ProQuest) data and employs modern NLP techniques including BERT-based sentiment analysis and TF-IDF based topic modeling.

## Project Goals

1. Process and extract news articles from major US newspapers stored in xml files and store them in csv tables (2000 articles in each table)
2. Classify articles as economic or non-economic using a a logistic regression model (under src/topic_modeling/is_economic_model) and select economic articles for the following steps 
3. Extract keywords from classified economic articles using tf idf method
4. Analyze sentiment score for classified economic articles
5. Create a structured dataset for further analysis

## Directory Structure

### `/corpuses/`
Contains XML files from major news sources including:
- Chicago Tribune
- Los Angeles Times
- Newyork Times
- The Washington Post
- USA Today

### `/data/`
- `/file_names/`: Lists of files from each corpus selected for specific tasks
- `/processed/classic_bert/results/`
- `/processed/results/`: Contains processed data with rich details (full articles) #TODO will add here a new folder for FinBert 
- `/processed/results_to_export/`: Contains processed data without raw content that can be exported

### `/logs/`
Contains execution logs that track sentiment analysis runs and other processing activities.

### `/notebooks/`
- `/run/`: Main execution notebooks used to run functions (preferred over pipline.py since TDM runs in Jupyter)
- `/experiments/`: Test notebooks for trying functions before running
- `/check_results/`: Notebooks for verifying processed results
- `/visualization/`: Notebooks for data visualization and reporting (including text_view to look at a specific article)

### `/src/`
Core source code:
- `config.py`: Configures global variables and folder paths
- `main.py`: Entry point for processing (limited use in TDM's Jupyter environment)
- `logger.py`: Tracks function execution to enable process resumption after failures
- `/data_utils/`: Data processing utilities
  - `tdm_parser.py`: Parser for TDM XML files
  - `xml_to_df.py`: Converts XML to DataFrame format
  - `properties_modifier.py`: Modifies CSV files with sentiment or topic data
- `/sentiment/`: Sentiment analysis components
  - `/sentiment_model/`: BERT sentiment models stored locally
  - `/salience_index/`: Weighted BERT sentiment model
- `topic_modeling/`: Core topic‑modeling modules  
  - `tf_idf_model/`: Extracts article tags using TF‑IDF  
    ```txt
    tfidf(t, d, D) = tf(t, d) * log(|D| / |{d' in D : t in d'}|)

    where:
      t            = term
      d            = document
      D            = corpus (set of all documents)
      tf(t, d)     = term frequency of t in d
    ```  
  - `is_economic_model/`: Logistic‑regression classifier for economic articles  
    - **Labels:** predefined sections  
    - **Features:** TF‑IDF–vectorized text  


## Key Files and Entry Points

### Configuration
- `src/config.py`: Central configuration of paths and settings

### Data Processing
- `src/data_utils/tdm_parser.py`: Handles TDM Studio XML format
- `src/data_utils/xml_to_df.py`: Processes XML data into DataFrames

### Analysis Components
- `src/sentiment/sentiment_model/sentiment_score.py`: BERT-based sentiment analyzer
- `src/topic_modeling/tf_idf_model/keyword_extractor.py`: TF-IDF keyword extraction
- `src/topic_modeling/is_economic_model/train_model.py`: Economic content classifier

### Main Entry Points
- `notebooks/run/run_title_sentiment.ipynb`: Runs sentiment analysis
- `notebooks/run/run_tf_idf_tags.ipynb`: Generates topic tags

## The `directory_map.txt` File

The `directory_map.txt` file in the src directory provides a comprehensive overview of the project structure and documents the purpose of each component. It serves as a navigation guide for understanding the organization of the codebase.

## TDM Studio Integration

This project is specifically designed to work with TDM Studio (ProQuest):
- Uses XML files from TDM Studio as input data
- Contains parsers specifically for TDM XML format
- Acknowledges TDM Studio's Jupyter environment constraints
- Optimized for TDM Studio's virtual machine setup
- fited for python version 10 

## Constraints and Limitations

1. **TDM Studio Environment**: The project is designed to run within TDM Studio's Jupyter environment, which may have limitations on package installations and system operations.
2. **Data Size**: The XML files can be large, requiring chunked processing.
3. **Proprietary Data**: The news articles are from proprietary sources and should be treated accordingly.
4. **Model Size**: The BERT models are large and may require significant resources.

## Setup and Running Instructions

### Requirements
- Python 3.7-10.9 
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- BeautifulSoup
- NLTK
- Torch

### Installation #TODO
1. Clone the repository:
   ```bash
   git clone https://github.com/<user>/tdm-sentiment.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Execution #TODO
Run the notebooks under `notebooks/run/` to process the data and calculate
sentiment scores, or execute:
```bash
python src/main.py
```
for scripted runs.

### Command-Line Pipeline
`src/pipeline.py` exposes the main steps of the project as a simple command line interface. The default paths are read from `src/config.py` but can be overridden.

Example:
```bash
python src/pipeline.py --corpus-dir Newyork20042023 \
    --steps xml_to_df,title_sentiment_prob
```
This runs XML extraction and sentiment scoring for the specified corpus using the configured directories.

With these steps you can reproduce the entire pipeline.
