import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import numpy as np
import evaluate
import argparse
#obviously, secret_token.py is not included in this repository (in gitignore)
# from secret_token import token
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from clean_utils import get_special_tokens
def train(input_csv: str, checkpoint: str = None):

    # with open(category_file, 'r') as f:
    #     categories = f.readlines()
    #     num_labels = len(categories)
    
    # dataset = load_dataset('csv', data_files=input_csv, split='train')
    # dataset = dataset.train_test_split(test_size=0.2)
    dataset = load_dataset("webis/tldr-17", split="train")
    dataset = dataset.train_test_split(test_size=0.05)
    random_train = dataset["train"].shuffle(seed=42).select(range(57000))
    random_test = dataset["test"].shuffle(seed=42).select(range(3000))

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    prev = "Summarize: "
    def tokenize_function(examples):
        # print(examples)
        inputs = [prev + doc for doc in examples["content"]]
        model_inputs = tokenizer(inputs, padding = "max_length", truncation=True)
        summary = tokenizer(examples["summary"], padding = "max_length", truncation=True)
        model_inputs["labels"] = subjects["input_ids"]
        return model_inputs
    

    tokenized_train = random_train.map(tokenize_function, batched=True)
    tokenized_test = random_test.map(tokenize_function, batched=True)

    train_dataset = tokenized_train
    eval_dataset = tokenized_test

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    # model.resize_token_embeddings(len(tokenizer))

    batch_size = 8
    num_train_epochs = 2

    logging_steps = len(train_dataset) // batch_size

    args = Seq2SeqTrainingArguments(
        output_dir=f"model-finetuned",
        evaluation_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        push_to_hub=False,
        
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        # Decode generated summaries into text
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Decode reference summaries into text
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # ROUGE expects a newline after each sentence
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        # Compute ROUGE scores
        rouge_score = evaluate.load("rouge")
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_dataset = train_dataset.remove_columns(dataset["train"].column_names)
    eval_dataset = eval_dataset.remove_columns(dataset["test"].column_names)
    # features = [train_dataset[i] for i in range(2)]
    # data_collator(features)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.eval()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()
    train(args.input_csv, args.checkpoint)

if __name__ == '__main__':
    main()
# def finetune(dataset):

#     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
#     model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")


#     # inputs = tokenizer([pair['document'] for pair in dataset], padding=True, truncation=True, return_tensors="pt")
#     # labels = tokenizer([pair['summary'] for pair in dataset], padding=True, truncation=True, return_tensors="pt")['input_ids']

#     # Set up training parameters
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Training loop
#     model.train()
#     model.to(device)
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         inputs.to(device)
#         labels.to(device)
#         outputs = model(**inputs, labels=labels)
#         loss = outputs.loss
#         loss.backward()
#         optimizer.step()
#         print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
#     return model
# def run(dataset, finetuned):
#     if finetuned:
#         summarizer = pipeline("summarization", model = "fine_tuned_summarizer", tokenizer="fine_tuned_summarizer")
#     else:
#         summarizer = pipeline("summarization")
    
#     for text in dataset:
#         classifier("text")

# finetune = True
# if finetune:
#     model = finetune(dataset)
#     model.save_pretrained("fine_tuned_summarizer")
# run(dataset, finetune)
