import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str, help='Input csv file')
    args = parser.parse_args()
    input_csv = args.input_csv
    df = pd.read_csv(input_csv)
    print("Label Distribution:")
    print(df['label'].value_counts())

if __name__ == '__main__':
    main()