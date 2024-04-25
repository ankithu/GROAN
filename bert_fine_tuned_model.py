import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import evaluate
import argparse
#obviously, secret_token.py is not included in this repository (in gitignore)
from secret_token import token
import torch
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight
from clean_utils import get_special_tokens

# class WeightedTrainer(Trainer):
#     def __init__(self, *args, class_weights=None, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.class_weights = class_weights

#     def compute_loss(self, model, inputs, return_outputs=False):
#         labels = inputs.get("labels")
#         outputs = model(**inputs)
#         logits = outputs.get('logits')
#         if self.class_weights is not None:
#             loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(self.model.device))
#             loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
#         else:
#             loss = F.cross_entropy(logits, labels)
#         return (loss, outputs) if return_outputs else loss

# def compute_class_weights(dataset):
#     # Extract all labels from the dataset
#     labels = dataset['label']

#     # Calculate label frequencies
#     unique, counts = np.unique(labels, return_counts=True)
#     label_counts = dict(zip(unique, counts))
#     print("Label Counts:", label_counts)

#     # Compute class weights
#     class_weights = compute_class_weight(
#         class_weight='balanced',
#         classes=np.unique(labels),
#         y=labels
#     )

#     # Convert the class weights to a tensor
#     class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
#     print("Class Weights Tensor:", class_weights_tensor)
#     return class_weights_tensor


def train(input_csv: str, category_file: str, checkpoint: str = None):
    
    with open(category_file, 'r') as f:
        categories = f.readlines()
        num_labels = len(categories)
    
    dataset = load_dataset('csv', data_files=input_csv, split='train')
    dataset = dataset.train_test_split(test_size=0.2)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    special_tokens = get_special_tokens()
    tokenizer.add_tokens(special_tokens)
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_dataset["train"].shuffle(seed=42).select(range(1000))
    eval_dataset = tokenized_dataset["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=num_labels, token=token)
    model.resize_token_embeddings(len(tokenizer))
    training_args = TrainingArguments(output_dir="test_trainer", 
                                      evaluation_strategy="epoch",
                                      save_strategy="epoch",
                                      per_device_train_batch_size=8,
                                      per_device_eval_batch_size=8,
                                      num_train_epochs=3,
                                      )
    metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels, average="micro")
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    #class_weights_tensor = compute_class_weights(train_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    # trainer = WeightedTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=eval_dataset,
    #     compute_metrics=compute_metrics,
    #     class_weights=class_weights_tensor
    # )

    trainer.train(resume_from_checkpoint=checkpoint)

#seperate function so that it can be called from other scripts/colab notebooks
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--category_file', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    args = parser.parse_args()
    train(args.input_csv, args.category_file, args.checkpoint)

if __name__ == '__main__':
    main()