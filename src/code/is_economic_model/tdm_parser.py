PROPERTY_TAGS = ['GOID', 'SortTitle', 
                'NumericDate', 'Language',
                'DocSection', 
                'mstar', 'GenSubjTerm']
    
PROPERTY_NAMES = ['GOID', 'Publisher',  
                 'Date', 'Language', 
                 'Section', 
                 'Type', 'Tags']





from bs4 import BeautifulSoup

class TdmXmlParser:
    PROPERTY_TAGS = [
        'GOID', 'SortTitle', 'Title', 'NumericDate', 'Language', 'StartPage',
        'DocSection', 'mstar', 'DocEdition', 'GenSubjTerm', 'CompanyName',
        'Personal', 'LastName', 'FirstName', 'LexileScore'
    ]
    PROPERTY_NAMES = [
        'GOID', 'Publisher', 'Title', 'Date', 'Language', 'Page', 'Section',
        'Type', 'Edition', 'Tags', 'Company Name', 'Personal', 'Author Last Name',
        'Author First Name', 'Lexile Score'
    ]

    def get_properties(self):
        """Retrieve the property tags and names."""
        return self.PROPERTY_TAGS, self.PROPERTY_NAMES

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


    # Modified function for extracting valid paragraphs from soup
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

    def get_xml_to_dict(self, soup, text=True, property_tags=None, property_names=None):
        """Convert XML content to a dictionary based on predefined tags."""
        if not soup:
            return {}

        # Adjusted to find 'Text' tag instead of 'RECORD'
        text_tag = soup.find('Text')
        if not text_tag:
            return {}

        # Use copies to avoid modifying the class variables
        if property_tags is None:
            property_tags = self.PROPERTY_TAGS.copy()
        if property_names is None:
            property_names = self.PROPERTY_NAMES.copy()

        content_dict = {}

        # Handle 'WordCount' if it is in property_tags
        if 'WordCount' in property_tags:
            index = property_tags.index('WordCount')
            name = property_names[index]
            # Extract the 'WordCount' attribute from the 'Text' tag
            word_count = text_tag.get('WordCount', None)
            content_dict[name] = word_count
            # Remove 'WordCount' from property_tags and property_names
            property_tags.pop(index)
            property_names.pop(index)

        # Continue processing the remaining tags
        for tag, name in zip(property_tags, property_names):
            prop = soup.find(tag)
            content_dict[name] = prop.get_text(strip=True).lower() if prop else None

        if text:
            # Assuming 'get_art_text' is a method to extract and clean text content
            text_content = self.get_art_text(text_tag)
            if text_content:
                content_dict['Text'] = text_content

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
