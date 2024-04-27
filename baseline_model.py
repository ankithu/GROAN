import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score


def get_distribution_from_file(file_path):
    category_distribution = []
    categories = []
    with open(file_path, 'r') as f:
        for line in f:
            category_name, category_freq = line.split(',')
            category_distribution.append(float(category_freq))
            categories.append(category_name)
    
    category_distribution = np.array(category_distribution)
    #normalize the distribution
    category_distribution /= category_distribution.sum()
    return category_distribution, categories

def main():
    args = argparse.ArgumentParser()
    args.add_argument('dataset_path', type=str, help='Dataset file to predict on')
    args.add_argument('category_distribution_file', type=str, help='File containing the category distribution')
    args.add_argument('correct_distribution_file', type=str, help='File containing the correct distribution')
    args = args.parse_args()

    df = pd.read_csv(args.dataset_path)

    category_distribution, categories = get_distribution_from_file(args.category_distribution_file)
    

    predictions = np.random.choice(len(categories), len(df), p=category_distribution)

    category_to_index = {category: i for i, category in enumerate(categories)}

    # data_categories = df['category']
    # data_categories = data_categories.apply(lambda x: category_to_index[x]) 
    
    #generate fake correct categories based on the distribution
    correct_distribution, _ = get_distribution_from_file(args.correct_distribution_file)
    data_categories = np.random.choice(len(categories), len(df), p=correct_distribution)

    accuracy = accuracy_score(data_categories, predictions)
    f1 = f1_score(data_categories, predictions, average=None)
    f1_micro = f1_score(data_categories, predictions, average='micro')

    print(f'Accuracy: {accuracy}')
    print(f'F1 Micro: {f1_micro}')
    f1s = f1.tolist()
    f1s = [str(f1) for f1 in f1s]
    print(f'F1: [{','.join(f1s)}]')

if __name__ == '__main__':
    main()
    





