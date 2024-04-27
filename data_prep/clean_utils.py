import pandas as pd
from user import User
from typing import Optional
from bs4 import BeautifulSoup
import html
import re

def get_special_tokens():
    #[SEP] is already a special token in BERT, these are additional special tokens
    return ['[USER]', '[LINK]', '[EMAIL]', '[IMAGE]', '[QUOTED_TEXT]', '[EMPTY]', '[HTML]']

def preprocess_html_email(html_content):
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
        return f"[HTML] {text}"
    else:
        # Return non-HTML content as is or apply other preprocessing
        return html_content

def clean_text_col(df: pd.DataFrame, user: Optional[User] = None) -> pd.DataFrame:
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].fillna("[EMPTY]")
    if user is not None:
        for name in user.get_names():
            df['text'] = df['text'].str.replace(name, "[USER]")
    
    df['text'] = df['text'].apply(preprocess_html_email)
    df['text'] = df['text'].str.replace(r'^>.*$', '[QUOTED_TEXT]', regex=True, flags=re.M)
    #in the text column, replace all instances of [image: ...] with "[IMAGE]"
    df['text'] = df['text'].str.replace(r'\[image:.*?\]', '[IMAGE]', regex=True)
    #replace all instances of [cid: ...] with "[IMAGE]"
    df['text'] = df['text'].str.replace(r'\[cid:.*?\]', '[IMAGE]', regex=True)
    #replace all instances of <http...> with "[LINK]"
    df['text'] = df['text'].str.replace(r'<http.*?>', '[LINK]', regex=True)
    df['text'] = df['text'].str.replace(r'\(http.*?\)', '[LINK]', regex=True)
    df['text'] = df['text'].str.replace(r'\[http.*?\]', '[LINK]', regex=True)
    #replace all words that start with http with "[LINK]"
    df['text'] = df['text'].str.replace(r'\bhttps?://\S+', '[LINK]', regex=True)

    #repalce mailto:..com|edu|org|co with [EMAIL]
    df['text'] = df['text'].str.replace(r'mailto:.*?\.(com|edu|org|co)', '[EMAIL]', regex=True)
    #remove all &nbsp; and &zwnj; characters with a space
    df['text'] = df['text'].str.replace(r'&nbsp;', ' ', regex=True)
    df['text'] = df['text'].str.replace(r'&zwnj;', ' ', regex=True)
    #make sure that all text is on a single line
    df['text'] = df['text'].str.replace(r'\n', ' ', regex=True)
    df['text'] = df['text'].str.replace(r'\r', ' ', regex=True)
    return df

def clean_subject_col(df: pd.DataFrame, user: Optional[User] = None) -> pd.DataFrame:
    df['subject'] = df['subject'].str.lower()
    df['subject'] = df['subject'].fillna("[EMPTY]")
    if user is not None:
        for name in user.get_names():
            df['subject'] = df['subject'].str.replace(name, "[USER]")
    return df

def get_output_text_col(df: pd.DataFrame, forSummarization):
    if not forSummarization:
        return df['subject'] + ' [SEP] ' + df['text'] + ' [SEP] ' + df['from_name'] + ' [SEP] ' + df['from_email'] + ' [SEP] ' + df['to'] + ' [SEP] ' + df['gmail_category'] + ' [SEP] ' + df['spam'].astype(str)
    else:
        return df['text'] + ' [SEP] ' + df['from_name'] + ' [SEP] ' + df['from_email'] + ' [SEP] ' + df['to'] + ' [SEP] ' + df['gmail_category'] + ' [SEP] ' + df['spam'].astype(str)

