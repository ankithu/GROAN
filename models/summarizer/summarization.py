import pandas as pd

import nltk
import bitsandbytes 
nltk.download("punkt")
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
import numpy as np
import evaluate
import argparse

import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

def train(input_csv: str, checkpoint: str = None):
    dataset = load_dataset("webis/tldr-17", split="train")
    dataset = dataset.train_test_split(test_size=0.05)
    random_train = dataset["train"].shuffle(seed=42).select(range(57000))
    random_test = dataset["test"].shuffle(seed=42).select(range(3000))

    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    prev = "Summarize: "
    def tokenize_function(examples):
        inputs = [prev + doc for doc in examples["content"]]
        model_inputs = tokenizer(inputs, padding = "max_length", max_length = 1024, truncation=True)
        summary = tokenizer(examples["summary"], padding = "max_length", max_length=128, truncation=True)
        model_inputs["labels"] = summary["input_ids"]
        return model_inputs
    

    tokenized_train = random_train.map(tokenize_function, batched=True)
    tokenized_test = random_test.map(tokenize_function, batched=True)

    train_dataset = tokenized_train
    eval_dataset = tokenized_test

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

    batch_size = 8
    num_train_epochs = 2

    logging_steps = len(train_dataset) // batch_size

    args = Seq2SeqTrainingArguments(
        output_dir=f"model-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5.6e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=num_train_epochs,
        predict_with_generate=True,
        logging_steps=logging_steps,
        push_to_hub=False,
        fp16 = True,
        optim="adamw_bnb_8bit" 
        
    )

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        
        rouge_score = evaluate.load("rouge")
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        # Extract the median scores
        result = {key: value * 100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}
    
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    train_dataset = train_dataset.remove_columns(dataset["train"].column_names)
    eval_dataset = eval_dataset.remove_columns(dataset["test"].column_names)
    
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
