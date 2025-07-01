# TDM Sentiment Analysis Project

## Project Overview

This project implements a comprehensive pipeline for analyzing sentiment and topics in news articles from major US newspapers, with a focus on financial and economic content. It leverages TDM Studio (ProQuest) data and employs modern NLP techniques including BERT-based sentiment analysis and TF-IDF based topic modeling.

## Project Goals

1. Process and analyze news articles from major US newspapers
2. Identify sentiment trends in financial/economic news
3. Extract key topics and keywords from news articles
4. Classify articles as economic or non-economic
5. Create a structured dataset for further analysis

## Directory Structure

### `/corpuses/`
Contains XML files from five major news sources:
- ChicagoTribune
- LosAngelesTimesDavid
- Newyork20042023
- TheWashingtonPostDavid
- USATodayDavid

### `/data/`
- `/file_names/`: Lists of files from each corpus selected for specific tasks
- `/processed/results/`: Contains processed data with rich details (full articles)
- `/processed/results_to_export/`: Contains processed data without raw content that can be exported

### `/logs/`
Contains execution logs that track sentiment analysis runs and other processing activities.

### `/notebooks/`
- `/run/`: Main execution notebooks used to run functions (preferred over main.py since TDM runs in Jupyter)
- `/experiments/`: Test notebooks for trying functions before running
- `/check_results/`: Notebooks for verifying processed results
- `/visualization/`: Notebooks for data visualization and reporting

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
- `/topic_modeling/`: Topic modeling components
  - `/tf_idf_model/`: TF-IDF model for extracting article tags
  - `/is_economic_model/`: Classifier for economic articles

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

## Questions Claude Can Help With

Claude can assist with:
1. Understanding the architecture and data flow
2. Explaining NLP techniques used (BERT, TF-IDF)
3. Adding new features or analysis components
4. Debugging processing issues
5. Optimizing code for performance
6. Extracting specific data from the processed results
7. Creating visualizations of sentiment trends
8. Extending the project to analyze new data sources

## Constraints and Limitations

1. **TDM Studio Environment**: The project is designed to run within TDM Studio's Jupyter environment, which may have limitations on package installations and system operations.
2. **Data Size**: The XML files can be large, requiring chunked processing.
3. **Proprietary Data**: The news articles are from proprietary sources and should be treated accordingly.
4. **Model Size**: The BERT models are large and may require significant resources.

## Setup and Running Instructions

### Requirements
- Python 3.7+
- PyTorch
- Transformers
- Scikit-learn
- Pandas
- BeautifulSoup
- NLTK

### Running the Analysis Pipeline

1. **Configure paths**:
   - Modify `src/config.py` to point to the correct directories

2. **Convert XML to DataFrames**:
   - Run `notebooks/run/run.ipynb`

3. **Perform Sentiment Analysis**:
   - Run `notebooks/run/run_title_sentiment.ipynb`

4. **Extract Keywords**:
   - Run `notebooks/run/run_tf_idf_tags.ipynb`

5. **Visualize Results**:
   - Use notebooks in `notebooks/visualization/`

### Working with Results
Processed results are stored in:
- `data/processed/results/[Newspaper]/chunk_0_data.csv`

## Data Flow

1. XML files from TDM Studio → `tdm_parser.py` → Structured data
2. Structured data → `xml_to_df.py` → Pandas DataFrames
3. DataFrames → Sentiment Analysis → Scored articles
4. DataFrames → Topic Modeling → Tagged articles
5. Tagged and scored articles → Final dataset for analysis

## Future Development Opportunities

1. Integrate more advanced language models
2. Add entity recognition for companies, people, and locations
3. Develop temporal analysis of sentiment trends
4. Implement cross-newspaper comparison tools
5. Add automated report generation