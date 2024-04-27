import mailbox
import pandas as pd
import uuid

def decode_payload(payload):
    try:
        # First, try decoding with utf-8
        return payload.decode('utf-8')
    except UnicodeDecodeError:
        # If utf-8 decoding fails, ignore undecodable bytes
        return payload.decode('utf-8', 'ignore')

def get_email_content(message):
    # Initialize variables to store different parts of the email
    text_content = None
    html_content = None

    # Check if the message is multipart
    if message.is_multipart():
        # Iterate over each part of the email
        for part in message.walk():
            # Get the content type of the part
            content_type = part.get_content_type()
            # Get the content disposition
            content_disposition = str(part.get("Content-Disposition"))

            # Look for text/plain parts, but ignore attachments
            if content_type == 'text/plain' and 'attachment' not in content_disposition:
                payload = part.get_payload(decode=True)
                text_content = decode_payload(payload)
            # elif content_type == 'text/html' and 'attachment' not in content_disposition:
            #     payload = part.get_payload(decode=True)
            #     html_content = decode_payload(payload)
    # If the message is not multipart, simply get its payload
    else:
        payload = message.get_payload(decode=True)
        text_content = decode_payload(payload)


    return text_content

def get_row_from_message(message):
    subject = message.get('subject', '')
    sender = message.get('from', '')
    recipient = message.get('to', '')
    date = message.get('date', '')
    labels = message.get('X-Gmail-Labels', '')
    spam = 'Spam' in labels
    category = 'None'
    for label in labels.split(','):
        if label.startswith('Category'):
            category = label.split(' ')[1]
            break
    text_content = get_email_content(message)
    message_id = str(uuid.uuid4())
    return {'message_id': message_id, 'subject': subject, 'from': sender, 'to': recipient,'date': date, 'labels': labels, 'text': text_content, 'spam': spam, 'gmail_category': category, 'labeled': False}


def mbox_to_dataframe_generator(mbox_path, batch_size=100):
    mbox = mailbox.mbox(mbox_path)
    rows = []
    for message in mbox:
        row = get_row_from_message(message)
        rows.append(row)
        # Yield a DataFrame when reaching the batch size
        if len(rows) == batch_size:
            yield pd.DataFrame(rows)
            rows = []
    # Yield any remaining messages as a DataFrame
    if rows:
        yield pd.DataFrame(rows)

def process_mbox_to_csv(mbox_path, csv_path):
    first_batch = True
    for df_batch in mbox_to_dataframe_generator(mbox_path, batch_size=20000):
        # Write the first batch with headers, then append without headers
        df_batch.to_csv(csv_path, mode='a', index=False, header=first_batch)
        first_batch = False

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('mbox_path', type=str, help='Path to the mbox file')
parser.add_argument('csv_path', type=str, help='Path to the output CSV file')
args = parser.parse_args()
mbox_path = args.mbox_path
csv_path = args.csv_path
process_mbox_to_csv(mbox_path, csv_path)