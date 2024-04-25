import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    #takes in variable number of arguments for all the csv files to combine
    parser.add_argument('csv_files', type=str, nargs='+', help='Input csv files')
    parser.add_argument('output_path', type=str, help='Output csv file')
    args = parser.parse_args()
    input_csvs = args.csv_files
    output_path = args.output_path
    #combine all csv files into one
    combined_csv = pd.concat([pd.read_csv(csv) for csv in input_csvs])
    combined_csv.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    main()