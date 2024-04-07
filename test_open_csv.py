import pandas as pd

df = pd.read_csv('Mail/mail.csv')

print(len(df))

#remove all nonstring and empty strings from the text column
df = df[df['text'].apply(lambda x: isinstance(x, str))]

print(len(df))

with_piazza = df[df['from'].str.contains('no-reply@piazza.com', case=False)]

sender_email_addresses = df['from'].str.extract(r'<(.*?)>')

#get top k senders
k = 50

#first make frequency table
frequency_table = sender_email_addresses.value_counts()
#now get the top k senders
top_k_senders = frequency_table.head(k)

print("Top k senders")
print(top_k_senders)

receiver_email_addresses = df['to'].str.extract(r'<(.*?)>')
#first make frequency table
frequency_table = receiver_email_addresses.value_counts()
#now get the top k receivers
top_k_receivers = frequency_table.head(k)

print("Top k receivers")
print(top_k_receivers)

total_message_tokens = df['text'].apply(lambda x: len(x.split()))
print("Total message tokens")
print(total_message_tokens)
print("Sum: ", total_message_tokens.sum())



# for sender in with_piazza['from'].head(50):
#     print(sender)
#     print("<<<<--------->>>>")

# print(len(with_piazza))