import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
from clean_utils import get_special_tokens

def main():
    #7th epoch is at test_trainer/checkpoint-11375
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help='Path to model')
    parser.add_argument('input_csv', type=str, help='Input csv file')
    parser.add_argument('num_inferences', type=int, help='Number of inferences to make')

    args = parser.parse_args()
    input_csv = args.input_csv
    num_inferences = args.num_inferences
    model_path = args.model_path

    #sample num_inferences rows from the input csv
    df = pd.read_csv(input_csv)
    df = df.sample(num_inferences)

    #load the model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = get_special_tokens()
    tokenizer.add_tokens(special_tokens)

    #tokenize the text
    inputs = tokenizer(df['text'].tolist(), padding=True, truncation=True, return_tensors='pt')

    #make inferences
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=1)

    #check the predictions against the actual labels
    correct = 0
    for i, pred in enumerate(predictions):
        if pred == df['label'].iloc[i]:
            correct += 1

   
    

    #print out the original text and prediction for the first 10
    for i in range(10):
        print("Text:", df['text'].iloc[i])
        print("Prediction:", predictions[i].item())
        print("Actual Label:", df['label'].iloc[i])
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    print(f"Accuracy: {correct/len(df)}")

    print("prediction distribution:")
    #convert the predictions to a pandas series
    predictions = pd.Series(predictions.numpy())
    print(predictions.value_counts())
    
if __name__ == '__main__':
    main()

