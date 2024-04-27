import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_csv', type=str, help='Input csv file')
    parser.add_argument('output_path', type=str, help='Output csv file')
    args = parser.parse_args()
    input_csv = args.input_csv
    df = pd.read_csv(input_csv)
    print("Label Distribution:")
    print(df['label'].value_counts())
    min_label = df['label'].value_counts().min()
    print("Min Label Count:", min_label)
    #undersample the majority classes
    grouped = df.groupby('label', as_index=False)
    df_undersampled = pd.concat([group.sample(n=min_label, random_state=1) for name, group in grouped])
    print("Label Distribution After Undersampling:")
    print(df_undersampled['label'].value_counts())
    df_undersampled.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    main()