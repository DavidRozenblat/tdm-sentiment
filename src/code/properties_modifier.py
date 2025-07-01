def update_log_file(result_folder_path, log_file_name, remaining_files, corpus_name):
    """
    Overwrite the log file with the updated list of remaining files to process.
    """
    log_file_path = LOGS_PATH / f'{corpus_name}_{log_file_name}_log_.txt'
    
    with open(log_file_path, 'w') as f:
        for file_name in remaining_files:
            f.write(file_name + '\n')


def title_sentiment(text, sentiment_analyzer):
    """
    Analyze the sentiment of a single title text.
    """
    return sentiment_analyzer.analyze_article_sentiment([text], method='bert')


def modify_csv_title_sentiment(result_folder_path, corpus_name):
    """
    Process CSV files by adding a 'title_sentiment' column using sentiment analysis.
    """
    log_file_names_list = get_logger_file_names('modify_sentiment_title', result_folder_path, corpus_name)
    if log_file_names_list is None:
        print("Modification aborted by the user.")
        return

    # We'll build a new list of unprocessed files as we go
    unprocessed_files = list(log_file_names_list)  # copy

    for i, file_name in enumerate(log_file_names_list):
        print(f"Processing chunk {i}: {file_name}")
        file_path = result_folder_path / file_name

        # Read the CSV file.
        df = pd.read_csv(file_path)

        # df['title_sentiment'] = df['title'].apply(lambda text: title_sentiment(text, sentiment_analyzer)) # TODO

        texts = df['title'].tolist()

        with tqdm(total=len(texts), desc=f"Processing chunk {i}", bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}") as pbar:
            results = Parallel(n_jobs=-1, backend='threading')(
                delayed(title_sentiment)(text, sentiment_analyzer) for text in texts
            )
            pbar.update(len(texts))

        # Assign results back to the DataFrame
        df['title_sentiment'] = results

        # Save the modified DataFrame back to CSV.
        df.to_csv(file_path, index=False)

        # Now remove this file from the list so we don't process it again
        if file_name in unprocessed_files:
            unprocessed_files.remove(file_name)

        # Update the log file after each file is processed
        update_log_file(result_folder_path, 'modify_sentiment_title', unprocessed_files, corpus_name)

    print("All files processed successfully!")


# Apply sentiment analysis on the 'texts' column using the global analyzer.
#df['title_sentiment'] = df['title'].apply(lambda text: title_sentiment(text, sentiment_analyzer))
        