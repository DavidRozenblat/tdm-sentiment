from bs4 import BeautifulSoup
import copy

class TdmXmlParser:
    TDM_PROPERTY_TAGS = [
        'GOID', 'SortTitle', 'NumericDate', 'mstar', 'DocSection', 
        'GenSubjTerm', 'StartPage', 'WordCount', 'Title', 'CompanyName', 
        'is_economic', 'bert_sentiment' 
        ] 
    PROPERTY_NAMES = [
        'goid', 'publisher', 'date', 'article_type', 'section', 
        'tdm_topic_tags', 'page', 'word_count', 'title', 'company_name', 
        'is_economic', 'bert_sentiment',
        ]

    def get_properties(self):
        """Retrieve the property tags and names."""
        return self.TDM_PROPERTY_TAGS, self.PROPERTY_NAMES

    def get_xml_soup(self, file_path):
        """Parse an XML file and return the BeautifulSoup object."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                xml_data = file.read()
            soup = BeautifulSoup(xml_data, 'xml')
            return soup
        except Exception as e:
            print(f"Error reading or parsing the file '{file_path}': {e}")
            return None


    # function for extracting valid paragraphs from soup
    def get_art_text(self, soup):
        """Extract and clean text from the soup object, returning a list of valid paragraphs."""
        if not soup:
            return None

        text_tag = soup.find('Text')
        if not text_tag:
            return None

        try:
            text_content = text_tag.get_text()
            html_soup = BeautifulSoup(text_content, 'html.parser')
            paragraphs = html_soup.find_all('p')
            
            # Initialize an empty list to store valid paragraphs
            valid_paragraphs = []
            
            for p in paragraphs:
                # Extract and clean the text
                cleaned_text = p.get_text(strip=True).lower()
                
                # Apply the filtering conditions
                if (
                    '@' not in p.text and
                    len(p.text.split()) >= 2 and
                    'credit:' not in p.text.lower()
                ):
                    valid_paragraphs.append(cleaned_text)
            
            return valid_paragraphs if valid_paragraphs else None

        except Exception as e:
            return f"Error extracting text: {e}"

        
    def get_word_count(self, soup):
        """Extract and clean text from the soup object, returning number of words in text."""
        if not soup:
            return None

        text_tag = soup.find('Text')
        if not text_tag:
            return None

        try:
            text_content = text_tag.get_text()
            html_soup = BeautifulSoup(text_content, 'html.parser')
            paragraphs = html_soup.find_all('p')
            
            # Initialize an empty list to store valid paragraphs
            word_count = 0
            
            for p in paragraphs:
                # Extract and clean the text
                cleaned_text = p.get_text(strip=True).lower()
                word_count += len(cleaned_text.split())
            
            return word_count if word_count != 0 else None

        except Exception as e:
            return f"Error extracting text: {e}"

     
    def soup_to_dict(self, soup, tdm_property_tags, property_names):
        """
        Convert XML content to a dictionary based on predefined tags.
        Handles 'Text' and 'WordCount' specially, then processes remaining tags.
        """
        # Make local copies so we don't mutate the original lists outside this function
        local_tags = tdm_property_tags.copy()
        local_names = property_names.copy()

        # Initialize the dictionary to hold our extracted data
        content_dict = {}

        # 1. Handle 'Text' tag (if it exists)
        if 'Text' in local_tags:
            # Find the index of 'Text'
            text_index = local_tags.index('Text')
            # Try to find the actual <Text> element in the soup
            text_tag = soup.find('Text')
            if text_tag:
                # Map the text to the correct property name
                content_dict[local_names[text_index]] = self.get_art_text(soup)
            else:
                # If the <Text> element isn't found, store None or empty string
                content_dict[local_names[text_index]] = None

            # Remove 'Text' from the lists so it won't be processed again
            local_tags.pop(text_index)
            local_names.pop(text_index)

        # 2. Handle 'WordCount' tag (if it exists)
        if 'WordCount' in local_tags:
            wcount_index = local_tags.index('WordCount')
            # Assign the wordcount to the correct property name
            content_dict[local_names[wcount_index]] = text_tag.get('WordCount', None)

            # Remove 'WordCount' from the lists
            local_tags.pop(wcount_index)
            local_names.pop(wcount_index)

        # 3. Process the remaining tags in a loop
        for tag, name in zip(local_tags, local_names):
            # Attempt to find the tag in the soup
            prop = soup.find(tag)
            # If found, get the text; if not, store None
            content_dict[name] = prop.get_text(strip=True) if prop else None

        return content_dict

    
    
    def soup_to_dict1(self, soup, tdm_property_tags, property_names):
        """Convert XML content to a dictionary based on predefined tags."""
        content_dict = {}
        local_property_names = property_names.copy()
        if not soup:
            return {}
        text_tag = soup.find('Text')
        if not text_tag:
            return {}
        # Handle 'Text' if it is in tdm_property_tags
        if 'Text' in tdm_property_tags:
            index = tdm_property_tags.index('Text')
            name = local_property_names[index]
            text_content = self.get_art_text(soup)
            if text_content:
                content_dict[name] = text_content
            # Remove 'Text' from tdm_property_tags and property_names
            tdm_property_tags.pop(index)
            local_property_names.pop(index)
        # Handle 'WordCount' if it is in tdm_property_tags
        if 'WordCount' in tdm_property_tags:
            index = tdm_property_tags.index('WordCount')
            name = local_property_names[index]
            # Extract the 'WordCount' attribute from the 'Text' tag
            word_count = text_tag.get('WordCount', None)
            content_dict[name] = word_count
            # Remove 'WordCount' from tdm_property_tags and property_names
            tdm_property_tags.pop(index)
            local_property_names.pop(index)
        # Continue processing the remaining tags
        for tag, name in zip(tdm_property_tags, local_property_names):
            prop = soup.find(tag)
            content_dict[name] = prop.get_text(strip=True).lower() if prop else None
       
        return content_dict
    


    def modify_tag(self, soup, tag_name, value):
        """Add or update a tag with a given value within the 'grades' tag."""
        existing_tag = soup.find(tag_name)
        if existing_tag:
            print(f"Element '{tag_name}' already exists. Updating value.")
            existing_tag.string = str(value)
        else:
            record = soup.find('RECORD')
            if not record:
                print("No 'RECORD' tag found.")
                return soup

            grades = record.find('grades')
            if not grades:
                grades = soup.new_tag('grades')
                record.append(grades)

            new_element = soup.new_tag(tag_name)
            new_element.string = str(value)
            grades.append(new_element)
        return soup

    def delete_tag(self, soup, tag_name):
        """Delete a specified tag from the soup object."""
        tag = soup.find(tag_name)
        if tag:
            tag.decompose()
        else:
            print(f"Tag '{tag_name}' not found.")
        return soup
