from datetime import datetime

class SalienceScorer:
    def __init__(self):
        """
        Initializes the SalienceScorer with predefined dataset page sizes and day scores.
        """
        self.dataset_page_size = {
            'washington post the': 3333
            # Add more datasets and their corresponding page sizes as needed
        }
        
        self.day_scores = {
            'sunday': 1.0,
            'saturday': 0.9,
            'monday': 0.8,
            'tuesday': 0.8,
            'wednesday': 0.8,
            'thursday': 0.8,
            'friday': 0.8
        }
    
    def get_week_day(self, date_input):
        """
        Returns the weekday name for a given date input.

        Args:
            date_input (datetime or str): Date input in datetime object or 'YYYY-MM-DD' string format.

        Returns:
            str or None: Weekday name (e.g., 'Monday') or None if invalid.
        """
        try:
            if isinstance(date_input, str):
                date_obj = datetime.strptime(date_input, '%Y-%m-%d')
            elif isinstance(date_input, datetime):
                date_obj = date_input
            else:
                raise ValueError("Invalid input type. Expected datetime or str.")
            
            return date_obj.strftime('%A')
        except ValueError as e:
            print(f"Error: {e}")
            return None

    def normalize_page(self, page):
        """
        Normalizes the page identifier.

        Args:
            page (str or int): Page identifier.

        Returns:
            str or None: Normalized page identifier or None if invalid.
        """
        if not isinstance(page, str):
            page = str(page)
        if page.strip() == '.':
            return None
        page = page.lower()
        parts = page.split('.')
        if len(parts) == 1:
            # Single part: ensure it's a single letter or number
            if parts[0].isdigit() or parts[0].isalpha():
                return parts[0].lstrip('0') if parts[0].isdigit() else parts[0]
            else:
                return None
        elif len(parts) == 2:
            # Two parts: remove leading zeros and sort
            part1, part2 = parts
            part1 = part1.lstrip('0') if part1.isdigit() else part1
            part2 = part2.lstrip('0') if part2.isdigit() else part2
            return '.'.join(sorted([part1, part2]))
        else:
            # More than two parts: invalid
            return None

    def get_page_size(self, dataset_name):
        """
        Retrieves the page size for a given dataset.

        Args:
            dataset_name (str): Name of the dataset.

        Returns:
            int or None: Page size if available, else None.
        """
        return self.dataset_page_size.get(dataset_name, None)

    def get_page_score(self, page_tag):
        """
        Calculates the salience score for a given newspaper page tag.

        Args:
            page_tag (str): The page tag in the format 'number.letter' or a standalone number.

        Raises:
            ValueError: If the `page_tag` format is invalid.

        Returns:
            float: The corresponding salience score based on the provided criteria.
        """
        try:
            split_tag = page_tag.split('.')
            if len(split_tag) == 2:
                number, letter = split_tag[0], split_tag[1]
                number = int(number)
                letter = letter.lower()

                if letter == 'a':
                    if number == 1:
                        return 1.0
                    elif number == 2:
                        return 0.9
                    elif number == 3:
                        return 0.8
                    else:
                        return 0.7
                elif letter in ['b', 'c']:
                    if number == 1:
                        return 0.9
                    elif number == 2:
                        return 0.8
                    elif number == 3:
                        return 0.7
                    else:
                        return 0.6
                else:
                    if number == 1:
                        return 0.8
                    elif number == 2:
                        return 0.7
                    elif number == 3:
                        return 0.6
                    else:
                        return 0.5
            else:
                return 0.5
        except Exception:
            raise ValueError(f'Invalid page_tag format. Page Tag: {page_tag}')

    def get_day_score(self, day):
        """
        Returns the salience score based on the day of the week.

        Args:
            day (str): The day of the week (e.g., 'Sunday', 'Monday').

        Raises:
            ValueError: If the input is not a valid day of the week.

        Returns:
            float: The corresponding score.
        """
        day = day.strip().lower()
        
        if day in self.day_scores:
            return self.day_scores[day]
        else:
            raise ValueError("Invalid day name. Please enter a valid day of the week.")

    def salience_score_helper(self, week_day, page_tag, article_size):
        """
        Computes the intermediate salience score based on day, page tag, and article size.

        Args:
            week_day (str): The weekday name.
            page_tag (str): The normalized page tag.
            article_size (float): Size of the article.

        Returns:
            float or None: The calculated intermediate score or None if day score is invalid.
        """
        day_score = self.get_day_score(week_day)
        page_score = self.get_page_score(page_tag)
        if not day_score:
            return None
        try:
            page_day_score = day_score * page_score
            score = round(page_day_score * (2 * article_size - article_size ** 2), 3)
            return score
        except ValueError as e:
            print(f"Invalid Page Tag: {page_tag} or Week Day: {week_day}, Error: {e}")
            return None

    def get_salience_score(self, date, page_tag, word_count, dataset_name):
        """
        Calculates the salience score for an article based on date, page, word count, and dataset.

        Args:
            date (str or datetime): Date of the article.
            page_tag (str or int): Page identifier.
            word_count (int): Word count of the article.
            dataset_name (str): Name of the dataset.

        Returns:
            float or None: Calculated salience score, or None if an error occurs.
        """
        try:
            week_day = self.get_week_day(date)
            if not week_day:
                return None
            normalized_page = self.normalize_page(page_tag)
            if not normalized_page:
                print(f"Normalization failed for page tag: {page_tag}")
                return None
            dataset_page_size = self.get_page_size(dataset_name)
            if not dataset_page_size:
                print(f"Unknown dataset name: {dataset_name}")
                return None
            article_size = float(word_count) / dataset_page_size
            if article_size > 1:
                article_size = 1
            return self.salience_score_helper(week_day, normalized_page, article_size)
        except Exception as e:
            print(f"Error calculating salience score: {e}")
            return None

# Example usage:
if __name__ == "__main__":
    scorer = SalienceScorer()
    
    days = ['Sunday', 'Saturday', 'Monday', 'Friday', 'Funday']
    tags = ['1.a', '2.b', '3.c', '4.a', '1.d', '5.e', '5']
    word_counts = [1500, 3000, 4500]  # Example word counts
    dataset_name = 'a'
    
    for day in days:
        for tag in tags:
            for wc in word_counts:
                score = scorer.get_salience_score(day, tag, wc, dataset_name)
                #print(f"Date: {day}, Page Tag: {tag}, Word Count: {wc}, Score: {score}")
                
a = scorer.normalize_page('.')
print(a)
