'''
Checks the users progress on how many emails in mail.csv are labeled.
'''
import pandas as pd

mail_csv = 'mail.csv'
output_csv = 'test_out1.csv'



mail_df = pd.read_csv('mail.csv', low_memory=False)

no_sender_df = mail_df[mail_df['from'].isnull()]

print(f"Number of emails with no sender: {len(no_sender_df)}")
print(f"Number of senders: {len(mail_df['from'].unique())}")


output_df = pd.read_csv('test_out1.csv')

print(f"Number of labeled senders: {len(output_df['from'].unique())}")

print(f"Number of labeled emails: {len(output_df)}")
print(f"Total number of emails: {len(mail_df)}")
print(f"Number of emails to label: {len(mail_df) - len(output_df)}")

print(f"Columns in output_df: {output_df.columns}")

#generate a map of 'from' to frequency of 'from' in mail_df
from_map = mail_df['from'].str.extract(r'<(.*?)>').value_counts().to_dict()
labeled_from_map = output_df['from'].str.extract(r'<(.*?)>').value_counts().to_dict()

#get the highest frequency entry in the from_map that does not have the same frequency in the labeled_from_map
for k, v in from_map.items():
    if labeled_from_map.get(k, 0) != v:
        print("Highest frequency entry in from_map that does not have the same frequency in the labeled_from_map:")
        print(f"From: {k}, Frequency: {v}, (Labeled Frequency: {labeled_from_map.get(k, 0)})")
        break



#get the number of entries in the from_map with frequency less than 6
x = 0
for i in range(max(from_map.values())+1):
    sender_count = len([k for k, v in from_map.items() if v == i])
    labeled_sender_count = len([k for k, v in labeled_from_map.items() if v == i and v == from_map[k]])
    x += (sender_count - labeled_sender_count) * i
    if sender_count - labeled_sender_count > 0:
        print(f"Number of senders to label with a frequency of {i}: {sender_count - labeled_sender_count}")


label_dist = output_df['category'].value_counts()

print("Label distribution: ")
print(label_dist)

print(x)