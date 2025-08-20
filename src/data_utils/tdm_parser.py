from bs4 import BeautifulSoup, NavigableString
from pathlib import Path

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

    def get_xml_soup(self, file_path: Path):
        """helper functon, Parse an XML file and return the BeautifulSoup object."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                xml_data = file.read()
            soup = BeautifulSoup(xml_data, 'xml')
            return soup
        except Exception as e:
            print(f"Error reading or parsing the file '{file_path}': {e}")
            return None


    def write_xml_soup(self, soup: BeautifulSoup, file_path: Path):
        """Write the BeautifulSoup object back to an XML file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(str(soup))
        except Exception as e:
            print(f"Error writing to the file '{file_path}': {e}")


    # function for extracting valid paragraphs from soup
    def get_art_text(self, soup: BeautifulSoup, return_str=True):
        """Extract and clean text from soup, returning a list of valid paragraphs if return_str = False."""
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
            valid_str = ''
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
                    valid_str += cleaned_text
            if return_str:
                return valid_str if valid_str else None
            if not return_str:
                return valid_paragraphs if valid_paragraphs else None

        except Exception as e:
            return f"Error extracting text: {e}"


    def get_word_count(self, soup: BeautifulSoup):
        """Extract and clean text from the xml file, returning number of words in text."""
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


    def soup_to_dict(self, soup: BeautifulSoup, tdm_property_tags: list, property_names: list):
        """
        Convert soup to a dictionary based on predefined tags.
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

    
    def get_tag_value(self, soup: BeautifulSoup, tag_name: str):
        """Retrieve the value of a specified tag from a soup."""
        tag = soup.find(tag_name)
        if tag:
            return tag.get_text(strip=True)
        else:
            print(f"Tag '{tag_name}' not found.")
            return None


    def modify_tag(self, soup: BeautifulSoup, tag_name: str, value: str, modify: bool = True):
        """Add or update a <tag_name> under <processed> with the given value."""
        # Ensure <processed> container exists
        record = soup.find("RECORD")
        processed = record.find('processed')
        if not processed:
            processed = soup.new_tag('processed')
            record.append(NavigableString('\n'))
            record.append(processed)
            record.append(NavigableString('\n'))

        # Look for existing tag inside <processed>
        existing = processed.find(tag_name)
        if existing:
            print(f"Element '{tag_name}' already exists.")
            if modify:
                existing.string = str(value)
            else:
                print(f"Not modifying '{tag_name}' as requested.")
                return
        else:
             # Create and append new tag under <processed> with surrounding newlines in one line
            new_tag = soup.new_tag(tag_name)
            new_tag.string = str(value)
            processed.extend([NavigableString('\n'), new_tag, NavigableString('\n')])

        return soup
        

    def delete_tag(self, soup: BeautifulSoup, tag_name: str):
        """Delete a specified tag from a xml file."""
        tag = soup.find(tag_name)
        if tag:
            tag.decompose()
        else:
            print(f"Tag '{tag_name}' not found.")
        return soup


if __name__ == '__main__':
    parser = TdmXmlParser()
    path = '/home/ec2-user/SageMaker/data/LosAngelesTimesDavid/422216858.xml'
    val = parser.get_tag_value(path=path, tag_name='tf_idf')
    #soup = parser.get_xml_soup(path)
    print(val)
