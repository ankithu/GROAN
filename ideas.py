'''

Privacy note: Make sure to not share the data with anyone. The data is the mbox file and the generated csv file.
We are including the python code to generate the csv file from the mbox file and the code to read and clean the csv file to
prep it for training, but if you wan't to train the model yourself you will need to download your own mbox file and run the
process_mbox_to_csv function on it.


When labelling data: we can take top senders and first label entire swaths of emails
from senders rather than labelling each email individually. This will save time and effort.

When tokenizing data: we can replace all instances of user's name or close strings
(e.g for fullname "Ankith Udupa", we could replace "Ankith", "Udupa", "Mr. Udupa", etc with <USER> token)
This will allow for more general tokenization and reduce the number of unique tokens in the dataset.
Also will help with privacy concerns and alleviate the concern that we are only training on two people's email data

Additional Preprocessing:

1. Remove all non-string and empty strings from the text column
2. Extract email addresses from the 'from' column

Training Data Structure:

Text Channel Inputs:
1. Sender Domain
2. Sender Prefix
3. Reciepient Domain
4. Reciepient Prefix
5. Email Subject
6. Email Text
7. Gmail Category

Tokenizer will generate a tokenized training dataset from the above data structure.

Model will be trained on the tokenized dataset

Output of the model will be 


'''