'''
Utility to coallesce the data by message_id,
takes a single csv file as input and modifies it to combine rows with the same message_id, 
favoring the later row in case of conflicts.
'''

import pandas as pd
import argparse

def coallesce_by_message_id(input_path):
    df = pd.read_csv(input_path)
    df = df.groupby('message_id').last().reset_index()
    df.to_csv(input_path, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input csv file')
    args = parser.parse_args()
    coallesce_by_message_id(args.input_path)

if __name__ == '__main__':
    main()