from pathlib import Path

class Logger:
    def __init__(self, log_dir: Path, log_file_name: str, corpus_name:str, initiate_file_list: list):
        # pull in LOGS_PATH _at runtime_ and inject it into the module namespace
        self.log_path = log_dir / corpus_name / f'{log_file_name}_log.txt'
        self.log_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists
        
        self.initiate_file_list = initiate_file_list # Store the list
        
        
    # --- Logging Functions ---
    def write_initial_file_list(self):
        """initiate log file names out of input file list"""
        # Write the names to the log file
        with self.log_path.open('w', encoding='utf-8') as f:
            for name in self.initiate_file_list:
                f.write(f"{name}\n")


    def read_file_names(self):
        """Read the log and returnn stripped lines as a list."""
        names = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    names.append(name)
        return names


    def get_file_names(self):
        """Ensure the log file exists and return its contents as a list of file names."""
        if not self.log_path.exists():
            # initiate the log file by writing file names from result_folder_path.
            self.write_initial_file_list()

        # Read file names into a list.
        file_names_lst = self.read_file_names()
        return file_names_lst


    def update_log_file(self, remaining_files: list):
        """Overwrite the log file with the updated list of remaining files to process."""
        with self.log_path.open('w', encoding='utf-8') as f:
            for file_name in remaining_files:
                f.write(file_name + '\n')






