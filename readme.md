# GROAN

Getting Rid of Annoying Notifications

EECS 595 Final Project

Core components for classifying and summarizing emails, in pursuit of a smarter email management system.

Note:
If attempting to run the model, you will unfortunately have to train it yourself on your own data (requiring labeling) as we don't have a secure/ethical way to publish our private email data and the model parameters are too big to host on github right now without LFS (future plan). Contact ankithu@umich.edu or hbaloch@umich.edu with any questions.

# Repo Structure

## data_prep/

Data preparation pipeline. All scripts to label training data and clean it.

Pipeline:

1. process_mbox.py to generate csv from mbox
2. build_dataset.py to label (described in more detail below)
3. clean_emails.py to clean labeled emails
4. clean_spam.py to clean additional spam csv
5. combine_csvs.py to combine cleaned csvs to form dataset
6. undersample.py to undersample based on minimum frequency label

## models/

### models/classifier

All code/notebooks for classifer training + inference

### models/summarizer

All code/notebooks for summarizer training + inference


# How to generate training data

## Get email data

first, create an mbox file with just your gmail data (lookup how to do this online) you should now have a file 'mail.mbox somehwere'. Put
it in this repo somewhere

## Setup env

Run `python -m venv env` to create your virtual env

Run `source env/bin/activate` to activate it

Then run `pip install -r requirements.txt` to get all required dependencies in the env

## Now generate your preliminary csv file 

Run:
`python3 process_mbox.py <MAILBOX FILE> <OUTPUT CSV>` 

## Now begin labelling (BETA, so go slow and save into multiple csvs that we can coallesce later if necessary)

## By sender

Run:

`python3 build_dataset.py <CSV_INPUT_PATH> <LABELED_DATA_OUTPUT_PATH> <CATEGORIES TXT FILE> --by_sender`

example:

`python3 build_dataset.py mail.csv test_out.csv categories --by_sender`


The --by_sender flag allows you to mass label emails by their sender domain. If you think that
some senders are even partially mixing categories, just skip them in the cli (by pressing S) when they come up. When you
are done labelling, type in 'Q' instead of labeling the one you are on and it will save the results.

## Indiviually

simply do not add the --by_sender flag to label emails indivually, for each email you will be shown the subject and sender and reciepient
Example:

`python3 build_dataset.py mail.csv test_out.csv categories`

## Important For Both Methods:

If you would like to continue where you left off you MUST append the --apend flag. If you don't, you'll get a warning about it,
and if you ignore the warning and continue, YOU WILL OVERWRITE EVERYTHING THAT YOU JUST LABELED. This is intentional in cases where you
actually want to do this but its important to know and thats why the warning is there.

example with append:

`python3 build_dataset.py mail.csv test_out.csv categories --by_sender --append`

DO NOT COMMIT YOUR GENERATED CSV FILE TO THE REPO, SEND IT SOME OTHER WAY

WHEN LABELLING, PLEASE ASK BEFORE ADDING A CATEGORY AND TRY TO AVOID REMOVING OR CHANGING THE NAMES OF ANY

To check your progress run:

`python3 progress_tracker.py <input_csv> <output_csv>`
