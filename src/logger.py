from pathlib import Path
import sys

class Logger:
    def __init__(self):
        # pull in LOGS_PATH _at runtime_ and inject it into the module namespace
        from config import LOGS_PATH
        globals()['LOGS_PATH'] = LOGS_PATH
        
    @staticmethod
    # --- Logging Functions ---
    def write_logger_file_names(result_folder_path, log_file_name, corpus_name):
        """
        Write the list of CSV file names in result_folder_path to the log file.
        """
        result_folder_path = Path(result_folder_path)
        log_file = LOGS_PATH / f'{corpus_name}_{log_file_name}_log_.txt'

        # Get all CSV file names in the result folder.
        file_list = [f.name for f in result_folder_path.glob('*.csv')]
        # WARNING: This sorting lambda assumes that file names have an underscore and an integer at index 1.
        # Ensure your files follow the expected naming pattern.
        file_list = sorted(file_list, key=lambda x: int(x.split('_')[1]))

        with open(log_file, 'w') as f:
            for name in file_list:
                f.write(name + '\n')
        return file_list

    @staticmethod
    def read_logger_file_names(log_file_name, corpus_name):
        """
        Read the log file and return a list of file names.
        """
        log_file = LOGS_PATH / f'{corpus_name}_{log_file_name}_log_.txt'
        file_names_lst = []
        with open(log_file, 'r') as f:
            for line in f:
                file_name = line.strip()
                if file_name:
                    file_names_lst.append(file_name)
        return file_names_lst

    @staticmethod
    def get_logger_file_names(log_file_name, result_folder_path, corpus_name):
        """
        Ensure the log file exists and return its contents as a list of file names.
        If the log file is empty, prompt the user for further action.
        """
        log_file = LOGS_PATH / f'{corpus_name}_{log_file_name}_log_.txt'

        if not log_file.exists():
            # Create the log file by writing CSV file names from result_folder_path.
            Logger.write_logger_file_names(result_folder_path, log_file_name, corpus_name)

        # Read file names into a list.
        file_names_lst = Logger.read_logger_file_names(log_file_name, corpus_name)

        return file_names_lst

    @staticmethod
    def update_log_file(result_folder_path, log_file_name, remaining_files, corpus_name):
        """
        Overwrite the log file with the updated list of remaining files to process.
        """
        log_file_path = LOGS_PATH / f'{corpus_name}_{log_file_name}_log_.txt'

        with open(log_file_path, 'w') as f:
            for file_name in remaining_files:
                f.write(file_name + '\n')







