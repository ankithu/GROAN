'''
This is the first stage of the pipeline. We will have a class that takes in a df and overrides the
call method to clean the input. Returning the cleaned df.
'''
from typing import List
from user import User
import argparse
import pandas as pd
import re
from bs4 import BeautifulSoup
import html

class Cleaner:
    def __init__(self, categories: List[str], inbox_user: User):
        self.categories = categories
        self.inbox_user = inbox_user

    
    def preprocess_html_email(self, html_content):
        # Check if content is likely HTML
        if '<html' in html_content.lower():
            # Parse HTML content
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts and styles
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            
            # Get text and normalize whitespace
            text = ' '.join(soup.stripped_strings)
            
            # Decode HTML entities
            text = html.unescape(text)
            #adding <HTML> tags to the beginning of the text, to indicate that it was originally HTML content
            return f"<HTML> {text}"
        else:
            # Return non-HTML content as is or apply other preprocessing
            return html_content

    def __call__(self, df: pd.DataFrame):
        '''
        This method will take in a df and clean it. It will return the cleaned df.

        Cleaning for this class involves:

        - Removing all rows from the df that contain a 'category' column that is not in the categories list.
        - Removing all rows from the df that contain None value in the 'from', 'to', 'date' columns.
        - Replacing all None values in the 'subject', 'text' columns with "<EMPTY>" token.
        - Replaces all instances of ways to address the inbox user with "<USER>" token.
        '''
        #remove all rows that contain a category that is not in the categories list
        df = df[df['category'].isin(self.categories)]
        #remove all rows that contain None value in the 'from', 'to', 'date' columns
        df = df.dropna(subset=['from', 'to', 'date'])
        #replace all None values in the 'subject', 'text' columns with "<EMPTY>" token
        df['subject'] = df['subject'].fillna("<EMPTY>")
        df['text'] = df['text'].fillna("<EMPTY>")
        #turn all text, subjects, from, and to lowercase
        df['text'] = df['text'].str.lower()
        df['subject'] = df['subject'].str.lower()
        df['from'] = df['from'].str.lower()
        df['to'] = df['to'].str.lower()

        #replace all instances of ways to address the inbox user with "<USER>" token
        for name in self.inbox_user.get_names():
            df['text'] = df['text'].str.replace(name, "<USER>")
            df['subject'] = df['subject'].str.replace(name, "<USER>")
            df['from'] = df['from'].str.replace(name, "<USER>")
            df['to'] = df['to'].str.replace(name, "<USER>")

        #for the from column, seperate email and name into two columns, from_email and from_name
        df['from_email'] = df['from'].str.extract(r'<(.*?)>')
        df['from_name'] = df['from'].str.split('<').str[0]

        #for the from_name column, remove any leading or trailing whitespace, and non-alphanumeric characters (except for spaces)
        df['from_name'] = df['from_name'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
        df['from_name'] = df['from_name'].str.strip()

        df['text'] = df['text'].apply(self.preprocess_html_email)
        df['text'] = df['text'].str.replace(r'^>.*$', '<QUOTED_TEXT>', regex=True, flags=re.M)
        #in the text column, replace all instances of [image: ...] with "<IMAGE>"
        df['text'] = df['text'].str.replace(r'\[image:.*?\]', '<IMAGE>', regex=True)
        #replace all instances of [cid: ...] with "<IMAGE>"
        df['text'] = df['text'].str.replace(r'\[cid:.*?\]', '<IMAGE>', regex=True)
        #replace all instances of <http...> with "<LINK>"
        df['text'] = df['text'].str.replace(r'<http.*?>', '<LINK>', regex=True)
        df['text'] = df['text'].str.replace(r'\(http.*?\)', '<LINK>', regex=True)
        df['text'] = df['text'].str.replace(r'\[http.*?\]', '<LINK>', regex=True)
        #replace all words that start with http with "<LINK>"
        df['text'] = df['text'].str.replace(r'\bhttps?://\S+', '<LINK>', regex=True)

        #repalce mailto:..com|edu|org|co with <EMAIL>
        df['text'] = df['text'].str.replace(r'mailto:.*?\.(com|edu|org|co)', '<EMAIL>', regex=True)
        #remove all &nbsp; and &zwnj; characters with a space
        df['text'] = df['text'].str.replace(r'&nbsp;', ' ', regex=True)
        df['text'] = df['text'].str.replace(r'&zwnj;', ' ', regex=True)
        #make sure that all text is on a single line
        df['text'] = df['text'].str.replace(r'\n', ' ', regex=True)
        for email in df['text'][100:105]:
            print(email)
        return df
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input csv file')
    parser.add_argument('category_path', type=str, help='File containing the categories')
    parser.add_argument('output_path', type=str, help='Output csv file')
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    with open(args.category_path, 'r') as f:
        categories = f.readlines()
    categories = [category.strip() for category in categories]
    ankith_user = User('Ankith', 'Udupa', 'ankithu', 'umich.edu', 'Mr.')
    cleaner = Cleaner(categories, ankith_user)
    cleaned_df = cleaner(df)
    cleaned_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()

