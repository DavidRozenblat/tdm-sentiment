from pathlib import Path

class Logger:
    def __init__(self, log_dir: Path, log_file_name: str, corpus_name:str, initiate_file_list: list):
        """Create a logger instance.

        Parameters
        ----------
        log_dir : Path
            Directory where log files are stored.
        log_file_name : str
            Base name of the log file.
        corpus_name : str
            Name of the corpus being processed.
        initiate_file_list : list
            List of all files that should eventually be processed.

        Notes
        -----
        The log file now tracks *processed* files.  Each time a file is
        successfully handled, its name is appended to this log.  Pending files
        are derived by subtracting the processed log from ``initiate_file_list``.
        """
        # pull in LOGS_PATH _at runtime_ and inject it into the module namespace
        self.log_path = log_dir / corpus_name / f'{log_file_name}_log.txt'
        self.log_path.parent.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

        self.initiate_file_list = initiate_file_list # Store the list
        
        
    # --- Logging Functions ---
    def write_initial_file_list(self):
        """Initialise the log file if it doesn't exist."""
        # Simply touch the file so subsequent appends work.
        self.log_path.open('a', encoding='utf-8').close()


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
        """Return the list of files that still need to be processed."""
        if not self.log_path.exists():
            # Ensure the log file exists before reading.
            self.write_initial_file_list()

        processed_files = set(self.read_file_names())
        # Determine which files have not yet been processed.
        file_names_lst = [name for name in self.initiate_file_list if name not in processed_files]
        return file_names_lst


    def update_log_file(self, processed_file: str):
        """Append a processed file name to the log.

        Parameters
        ----------
        processed_file : str
            The name of the file that has just been processed.
        """
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(processed_file + '\n')






