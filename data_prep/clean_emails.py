'''
This is the first stage of the pipeline. We will have a class that takes in a df and overrides the
call method to clean the input. Returning the cleaned df.
'''
from typing import List
from user import User
import argparse
import pandas as pd
from bs4 import BeautifulSoup
import html
from clean_utils import clean_text_col, get_output_text_col, clean_subject_col

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

    def __call__(self, df: pd.DataFrame, do_summarization):
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
        #turn all text, subjects, from, and to lowercase

        df = clean_text_col(df, self.inbox_user)
        df = clean_subject_col(df, self.inbox_user)

        df['from'] = df['from'].str.lower()
        df['to'] = df['to'].str.lower()
 
        df['gmail_category'] = df['gmail_category'].fillna("<EMPTY>")

        #replace all instances of ways to address the inbox user with "<USER>" token
        for name in self.inbox_user.get_names():
            df['from'] = df['from'].str.replace(name, "<USER>")
            df['to'] = df['to'].str.replace(name, "<USER>")

        #for the from column, seperate email and name into two columns, from_email and from_name
        df['from_email'] = df['from'].str.extract(r'<(.*?)>')
        df['from_email'] = df['from_email'].fillna("<EMPTY>")
        df['from_name'] = df['from'].str.split('<').str[0]

        #for the from_name column, remove any leading or trailing whitespace, and non-alphanumeric characters (except for spaces)
        df['from_name'] = df['from_name'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
        df['from_name'] = df['from_name'].str.strip()
        #create output df with only two columns, 'text' and 'label', text will have all relevant columns concatenated, seperated by SEP tokens
        output_df = pd.DataFrame()
        
        if not do_summarization:
            output_df['label'] = df['category'].apply(lambda x: self.categories.index(x))
        else:
            output_df['subject'] = df['subject']
        output_df['text'] = get_output_text_col(df, do_summarization)
        #the label column should be index of the category in the categories list
        return output_df
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input csv file')
    parser.add_argument('category_path', type=str, help='File containing the categories')
    parser.add_argument('user', type=str, help='User name (either "haider" or "ankith")')
    parser.add_argument('output_path', type=str, help='Output csv file')
    parser.add_argument('--summarization', action="store_true", help="Set the value to True if this argument is called")

    args = parser.parse_args()
    do_summarization = args.summarization
    df = pd.read_csv(args.input_path)
    with open(args.category_path, 'r') as f:
        categories = f.readlines()
    categories = [category.strip() for category in categories]
    ankith_user = User('Ankith', 'Udupa', 'ankithu', 'umich.edu', 'Mr.')
    haider_user = User('Haider', 'Baloch', 'hbaloch', 'umich.edu', 'Mr.')
    if args.user.lower() == 'haider':
        user = haider_user
    else:
        user = ankith_user
    cleaner = Cleaner(categories, user)
    cleaned_df = cleaner(df, do_summarization=do_summarization)
    cleaned_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()

