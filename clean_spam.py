import pandas as pd
from typing import List
from clean_utils import clean_text_col, get_output_text_col, clean_subject_col
import argparse

class SpamCleaner:
    def __init__(self, categories: List[str], names_file: str, emails_file: str):
        self.categories = categories
        self.categories = [category.lower() for category in self.categories]
        with open(names_file, 'r') as f:
            names = f.read().split(',')
            self.names = [name.strip() for name in names]
        with open(emails_file, 'r') as f:
            emails = f.read().split(',')
            self.emails = [email.strip() for email in emails]

    def __call__(self, df: pd.DataFrame):
        #spam dataset just has Subject, Message columns
        output_df = pd.DataFrame()
        #we only want to keep spam emails
        df = df[df['Spam/Ham'] == 'spam']

        #only keep 2000 spam emails
        df = df.head(2000)

        output_df['text'] = df['Message']
        output_df['subject'] = df['Subject']
        output_df = clean_text_col(output_df)
        output_df = clean_subject_col(output_df)
        output_df['label'] = df['Spam/Ham'].apply(lambda x: self.categories.index(x))

        #we need to add the other columns to construct the message, namely from_name, from_email, to, gmail_category, and spam
        #we don't actually have this information, so we will construct it reasonably
        #randomly select a name and email from the list
        output_df['from_name'] = [self.names[i % len(self.names)] for i in range(len(output_df))]
        output_df['from_email'] = [self.emails[i % len(self.emails)] for i in range(len(output_df))]

        #to will be either [USER] or a random email (but usually [USER])
        #adding 1 to the index to avoid the case where the email is the same as the from email
        output_df['to'] = ['[USER]' if i % 10 != 0 else self.emails[(i+1) % len(self.emails)] for i in range(len(output_df))]

        #gmail claims to correctly categorize spam 99.9% of them time so we will have spam be true 99.9% of the time
        output_df['spam'] = [True if i % 1000 != 0 else False for i in range(len(output_df))]
        output_df['gmail_category'] = '[EMPTY]'
        output_df['text'] = get_output_text_col(output_df)
        real_output_df = output_df[['text', 'label']]

        return real_output_df
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input csv file')
    parser.add_argument('category_file', type=str, help='File containing the categories')
    parser.add_argument('names_file', type=str, help='File containing the names')
    parser.add_argument('emails_file', type=str, help='File containing the emails')
    parser.add_argument('output_path', type=str, help='Output csv file')
    args = parser.parse_args()
    df = pd.read_csv(args.input_path)
    with open(args.category_file, 'r') as f:
        categories = f.readlines()
    categories = [category.strip() for category in categories]
    cleaner = SpamCleaner(categories, args.names_file, args.emails_file)
    cleaned_df = cleaner(df)
    cleaned_df.to_csv(args.output_path, index=False)


if __name__ == '__main__':
    main()






       