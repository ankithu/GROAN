# How to generate training data

## Get email data

first, create an mbox file with just your gmail data (lookup how to do this online) you should now have a file 'mail.mbox somehwere'. Put
it in this repo somewhere

## Setup env

Run `python -m venv env` to create your virtual env

Run `source env/bin/activate` to activate it

Then run `pip install -r requirements.txt` to get all required dependencies in the env

## Now generate your preliminary csv file

Go into process.py and scroll to the bottom.
There are two variables, mbox_path and csv_path. mbox_path is the path to your
mbox file, and csv path is the path to your output csv file. Change them to be correct for your usecase.

TODO change these to program args. 

Run:
`python3 process.py` 

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


