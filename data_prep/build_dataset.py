import argparse
import os
import pandas as pd
from typing import List, Optional

def convertable_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def label_by_sender(df: pd.DataFrame, input_df_path: str, categories: List[str], tgt_df_ref: Optional[pd.DataFrame], max_freq: Optional[int]) -> pd.DataFrame:
    #find senders already labeled in the target dataframe (if any one email from a sender is labeled, you won't be able to label it through --by_sender
    #since that would cause potential inconsistencies. So if you label any one email from a sender manually, you need to do the same for all senders from that domain)
    
    #get all the unlabled emails from the dataset
    unlabeled_df = df#[df['labeled'] == False]
    
    labeled_sender_emails = []
    if tgt_df_ref is not None:
        labeled_sender_emails = tgt_df_ref['from'].str.extract(r'<(.*?)>').value_counts()
    
    
    
    #sort all senders by frequency
    sender_emails = unlabeled_df['from'].str.extract(r'<(.*?)>')
    sender_frequency = sender_emails.value_counts()
    orig_df_cols = unlabeled_df.columns
    output_df = pd.DataFrame(columns=orig_df_cols)
    print(labeled_sender_emails['eecs-comm@umich.edu'], sender_frequency['eecs-comm@umich.edu'])
    for sender in sender_frequency.index:
        #if this sender is already labeled, skip it
        if max_freq is not None and sender_frequency[sender] > max_freq:
            continue
        relabeling = False
        if sender in labeled_sender_emails.index:
            # if 'no-reply@piazza.com' in sender:
            #     print("forcing to label piazza, even though it is already labeled partially")
            # elif 'no-reply@gradescope.com' in sender:
            #     print("forcing to label ankithu, even though it is already labeled partially")
            # elif 'notifications@instructure.com' in sender:
            #     print("forcing to label canvas, even though it is already labeled partially")
            # if 'coe-studentaffairs@umich.edu' in sender:
            #     print("forcing to label coe-studentaffairs, even though it is already labeled partially")
            # if 'akamil@umich.edu' in sender:
            #     print("forcing to label akamil, even though it is already labeled partially")
            if labeled_sender_emails[sender] != sender_frequency[sender]:
                relabeling = True
                print(f"RELABELING {sender} with {labeled_sender_emails[sender]} labeled emails and {sender_frequency[sender]} total emails")
            else:
                continue
        #some spacing
        print('\n\n')
        print('-----------------------------------')
        print(f"SENDER DOMAIN: {sender}, Frequency: {sender_frequency[sender]}")
        if relabeling:
            print("(RELABELING)")

        #start with printig top 5 full senders
        category = "MS:5"

        sender = sender[0]
        #figure out distribution of 'gmail_category' for this sender
        sender_df = unlabeled_df[unlabeled_df['from'].str.contains(sender, case=False, regex=False)]

        gmail_category_distribution = sender_df['gmail_category'].value_counts()
        sender_distribution = sender_df['from'].value_counts()

        if category == 'MS:5':     
            print("GMAIL generated category distribution: ")
            print(gmail_category_distribution)

        #get subjects of first 3 emails
        print("Subjects of first 3 emails: ")
        print(sender_df['subject'].head(3))

        first = category == 'MS:5'

        #if user wants to see more senders, show more senders until they are satisfied
        while first or not (category == 'S' or category == 'Q' or convertable_to_int(category)):
            
            if (not first) and (not convertable_to_int(category)) and (not category.startswith('MS:')):
                print("INVALID INPUT, TRY AGAIN")
            
            first = False
            k = int(category.split(':')[1])
            print(f"FULL Sender distribution (top {k}): ")
            print(sender_distribution.head(k))
            #ask user for category for this sender
            print("Categories: ")
            for i, c in enumerate(categories):
                print(f"\t{i+1}. {c}")
            new_category = input("Enter category for this sender (Q to stop, S to skip, 'MS:<NUM>' to change shown full sender distribution to <NUM> senders): ")
            if not (new_category == 'S' or new_category == 'Q' or new_category.startswith('MS:') or convertable_to_int(new_category)):
                print("INVALID INPUT, TRY AGAIN")
                continue
            else:
                category = new_category
            

        if category == 'Q':
            break
        if category == 'S':
            continue

        
        
        category = categories[int(category)-1]
        
        #generate rows for each email from this sender with label attached
        rows = []
        for _, row in sender_df.iterrows():
            row['labeled'] = True
            row['category'] = category
            rows.append(row)
        output_df = pd.concat([output_df, pd.DataFrame(rows)])

        #mark all emails from this sender as labeled in the original dataframe
        df.loc[df['from'].str.contains(sender, case=False, regex=False), 'labeled'] = True
    
    #overwrite the original dataframe with the labeled emails
    df.to_csv(input_df_path, index=False)
    return output_df



def label_by_message(df: pd.DataFrame, input_df_path: str, categories: List[str]) -> pd.DataFrame:
    #get all emails that are not labeled
    unlabeled_df = df[df['labeled'] == False]
    print(len(unlabeled_df))

    #start marching through the emails
    output_df = pd.DataFrame(columns=df.columns)
    for _, row in unlabeled_df.iterrows():
        print('\n\n')
        print('-----------------------------------')
        print(f"From: {row['from']}")
        print(f"Subject: {row['subject']}")
        print(f"Receipeint(s): {row['to']}")
        text = row['text']
        

        #print(f"Body: {row['text']}")
        print("Categories: ")
        for i, c in enumerate(categories):
            print(f"\t{i+1}. {c}")
        category = input("Enter category for this email (Q to stop, T for full text): ")
        
        
        if category == 'Q':
            break

        q = False

        while not convertable_to_int(category):
            if category != 'T':
                print("INVALID INPUT, TRY AGAIN")
            else:
                print(text)
            print("Categories: ")
            for i, c in enumerate(categories):
                print(f"\t{i+1}. {c}")
            category = input("Enter category for this email (Q to stop, T for full text): ")
            if category == 'Q':
                q = True
                break
        if q:
            break

        
        category = categories[int(category)-1]
        row['category'] = category
        row['labeled'] = True
        output_df = pd.concat([output_df, pd.DataFrame([row])])
        #mark the email as labeled in the original dataframe
        df.loc[df['message_id'] == row['message_id'], 'labeled'] = True
    
    #overwrite the original dataframe with the emails we labeled marked
    df.to_csv(input_df_path, index=False)

    #return the labeled emails
    return output_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', type=str, help='Dataset file')
    parser.add_argument('output_path', type=str, help='Output file')
    parser.add_argument('category_file', type=str, help='File with categories, each category on a new line')
    #parser.add_argument('uniqname', type=str, help='Uniqname of the person running this script')
    parser.add_argument('--by_sender', action='store_true', help='Group by sender')
    parser.add_argument('--append', action='store_true', help='Append to output file instead of overwriting')
    #optional argument for maximum frequency per sender to label
    parser.add_argument('--max_freq', type=int, help='Maximum frequency per sender to label')
    args = parser.parse_args()
    print(args.dataset_path)
    print(args.output_path)
    print(args.by_sender)
    print(args.append)
    print(args.category_file)
    print(args.max_freq)
    #print(args.uniqname)

    #get the categories
    with open(args.category_file, 'r') as f:
        categories = f.readlines()
    
    categories = [category.strip() for category in categories]
    print(categories)

    #check if the output file exists, if it does and append is not set, then warn the user
    if os.path.exists(args.output_path) and not args.append:
        cont = 'x'
        while cont != 'y' and cont != 'n':
            cont = input('Output file exists, use --append to append to the file. Do you want to continue? (y/n): ')
        if cont == 'n':
            return 1
    
    #read the dataset
    df = pd.read_csv(args.dataset_path, low_memory=False)


    if args.by_sender:
        out = label_by_sender(df, args.dataset_path, categories, pd.read_csv(args.output_path) if args.append else None, args.max_freq)
        out.to_csv(args.output_path, mode='a' if args.append else 'w', index=False)
    else:
        label_by_message(df, args.dataset_path, categories).to_csv(args.output_path, mode='a' if args.append else 'w', index=False)
        
        

if __name__ == '__main__':
    main()