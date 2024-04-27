from openai import OpenAI
from secret_token import open_ai_key
import pandas as pd
import time
from tqdm import tqdm
import json

model_name = "gpt-4-turbo-2024-04-09"

client = OpenAI(
    # This is the default and can be omitted
    api_key=open_ai_key,
)

index_to_categories = [
    "Class Update",
    "School-wide Update",
    "Career Update",
    "Club Update",
    "Personal Correspondence",
    "Company or Subscription",
    "News",
    "Spam"
]

def send_prompt(prompt):
    chat_completion = client.chat.completions.create(
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant designed to output classifications for emails in JSON format.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model_name,
    )

    return chat_completion

def get_example_prompts(df: pd.DataFrame, num_examples_per_class: int):
    example_prompts = []
    for label in df['label'].unique():
        label_df = df[df['label'] == label]
        for i in range(num_examples_per_class):
            text = label_df.iloc[i]['text']
            label = label_df.iloc[i]['label']
            as_dict = {
                "text": text,
                "label": label,
                "category": index_to_categories[label]
            }
            example = str(as_dict)
            example_prompts.append(example)
    return example_prompts


def get_category_prompts():
    as_list_of_dicts = [
        {"label": 0, "category": "Class Update", "description": "class specific information"},
        {"label": 1, "category": "School-wide Update", "description": "school-wide information or imilar school related mass email"},
        {"label": 2, "category": "Career Update", "description": "career related information or opportunities"},
        {"label": 3, "category": "Club Update", "description": "club related information or announcements"},
        {"label": 4, "category": "Personal Correspondence", "description": "personal email, directely to user"},
        {"label": 5, "category": "Company or Subscription", "description": "company or subscription update/promption"},
        {"label": 6, "category": "News", "description": "news"},
        {"label": 7, "category": "Spam", "description": "spam"}
    ]
    return str(as_list_of_dicts)
    

def build_prompt(email_text: str, train_df: pd.DataFrame, num_examples_per_class: int):
    """
    Builds a prompt asking the model to classify email_text based on examples 
    """
    prompt_prefix = ("You need to classify the following email. "
                    "Here is a list of categories formated as a list of JSON ojects. \n")
    category_prompts = get_category_prompts()
    transition = "\n Here are some examples of other emails that have already been classified, formatted as a list of JSON objects. \n"
    example_prompts = get_example_prompts(train_df, num_examples_per_class)
    example_prompts = "\n".join(example_prompts)
    email_prefix = "\n Please classify the following email in JSON format. Only include the label and category fields in the JSON object. \n"
    email_as_dict = {
        "text": email_text
    }
    email_text = str(email_as_dict)
    query = prompt_prefix + category_prompts + transition + example_prompts + email_prefix + email_text
    return query


def test_model(df: pd.DataFrame, total_inferences: int):
    per_label_tps = [0 for i in range(8)]
    per_label_fps = [0 for i in range(8)]
    per_label_fns = [0 for i in range(8)]

    #shuffle df
    df = df.sample(frac=1, random_state=42)

    

    train_df, test_df = df.iloc[:int(len(df) * 0.8)], df.iloc[int(len(df) * 0.8):]

    df = test_df.sample(total_inferences, random_state=42)

    print(df["label"].value_counts())

    finished_inferences = 0

    try:                
        for i in tqdm(range(total_inferences)):
            row = df.iloc[i]
            prompt = build_prompt(row['text'], train_df, 1)
            #print(prompt)
            start = time.time()
            response = send_prompt(prompt)
            end = time.time()
            #print(response)
            #output_json = .model_dump_json()['choices'][0]['message']['content']
            message_content = response.choices[0].message.content
            json_content = json.loads(message_content)
            output_label = json_content['label']
            actual_label = row['label']
            # print("Output Label:", output_label)
            # print("Actual Label:", actual_label)
            if output_label == actual_label:
                per_label_tps[output_label] += 1
            else:
                per_label_fps[output_label] += 1
                per_label_fns[actual_label] += 1

            #print("Time:", end-start)
            finished_inferences += 1
    except Exception as e:
        print(e)
        print("computing performance and dumping results anyways")
    
    print(f"Finished {finished_inferences} inferences")
    
    per_label_f1s = []
    for i in range(8):
        tp = per_label_tps[i]
        fp = per_label_fps[i]
        fn = per_label_fns[i]
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)
        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        per_label_f1s.append(f1)
        print(f"Label {i} F1: {f1}")
    print("(Macro) Average F1:", sum(per_label_f1s) / 8)

    micro_tp = sum(per_label_tps)
    micro_fp = sum(per_label_fps)
    micro_fn = sum(per_label_fns)

    micro_precision = micro_tp / (micro_tp + micro_fp + 1e-6)
    micro_recall = micro_tp / (micro_tp + micro_fn + 1e-6)
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-6)
    print(f"(Micro) Average F1: {micro_f1}")

    print(f"Overal Accuracy: {micro_tp / (micro_tp + micro_fp)}")

    #put performance into dataframe
    performance_df = pd.DataFrame()
    performance_df['label'] = list(range(8))
    performance_df['TP'] = per_label_tps
    performance_df['FP'] = per_label_fps
    performance_df['FN'] = per_label_fns
    performance_df['F1'] = per_label_f1s
    performance_df.to_csv('zero_shot_results/single-example.csv', index=False)

def main():
    df = pd.read_csv('undersampled_cleaned_all.csv')
    test_model(df, 500)

if __name__ == '__main__':
    main()
