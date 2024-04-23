'''
This is the first stage of the pipeline. We will have a class that takes in a df and overrides the
call method to clean the input. Returning the cleaned df.
'''
from typing import List
from user import User

class Cleaner:
    def __init__(self, categories: List[str], inbox_user: User):
        self.categories = categories
        self.inbox_user = inbox_user

    def __call__(self, df):
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
        df['subject'].fillna("<EMPTY>", inplace=True)
        df['text'].fillna("<EMPTY>", inplace=True)
        #replace all instances of ways to address the inbox user with "<USER>" token
        for name in self.inbox_user.get_names():
            df['text'] = df['text'].str.replace(name, "<USER>")
            df['subject'] = df['subject'].str.replace(name, "<USER>")
            df['from'] = df['from'].str.replace(name, "<USER>")
            df['to'] = df['to'].str.replace(name, "<USER>")
        return df
        