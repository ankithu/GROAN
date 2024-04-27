import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import pandas as pd
from datasets import load_metric
from clean_utils import get_special_tokens
import numpy as np
from datasets import load_dataset

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_metric = load_metric("f1")
    acc_metric = load_metric("accuracy")
    per_class_f1 = f1_metric.compute(predictions=predictions, references=labels, average=None)['f1']
    return {
        "f1": f1_metric.compute(predictions=predictions, references=labels, average="micro")['f1'],
        "accuracy": acc_metric.compute(predictions=predictions, references=labels)['accuracy'],
        "class_0_f1": per_class_f1[0],
        "class_1_f1": per_class_f1[1],
        "class_2_f1": per_class_f1[2],
        "class_3_f1": per_class_f1[3],
        "class_4_f1": per_class_f1[4],
        "class_5_f1": per_class_f1[5],
        "class_6_f1": per_class_f1[6],
        "class_7_f1": per_class_f1[7],
    }


dataset = load_dataset('csv', data_files='cleaned_all.csv', split='train')
dataset = dataset.train_test_split(test_size=0.05)
eval_dataset = dataset["test"]

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
special_tokens = get_special_tokens()
tokenizer.add_tokens(special_tokens)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)


results = []
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
checkpoints_dir = "model_params/classifier/undersampled_special_tokens_pretrained_bert"

for checkpoint in sorted(os.listdir(checkpoints_dir)):
    if checkpoint.startswith("checkpoint-"):
        model_path = os.path.join(checkpoints_dir, checkpoint)
        print(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)

        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir="temp_dir"),  # Temp dir for evaluation
            eval_dataset=tokenized_eval_dataset,
            compute_metrics=compute_metrics
        )

        eval_results = trainer.evaluate()
        print(eval_results)
        results.append({
            "checkpoint": checkpoint,
            "f1": eval_results["eval_f1"],
            "accuracy": eval_results["eval_accuracy"],
            "class_0_f1": eval_results["eval_class_0_f1"],
            "class_1_f1": eval_results["eval_class_1_f1"],
            "class_2_f1": eval_results["eval_class_2_f1"],
            "class_3_f1": eval_results["eval_class_3_f1"],
            "class_4_f1": eval_results["eval_class_4_f1"],
            "class_5_f1": eval_results["eval_class_5_f1"],
            "class_6_f1": eval_results["eval_class_6_f1"],
            "class_7_f1": eval_results["eval_class_7_f1"],
        })

# Convert results to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv("model_evaluation_results.csv", index=False)