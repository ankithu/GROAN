from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.add_tokens(['<SEP>', '<LINK>', '<IMAGE>', '[USER]'])

test_input = "This is a test input <SEP> [USER] says hi <SEP> There is a <LINK> in the document <SEP> And an <IMAGE>"

tokens = tokenizer(test_input, padding=True, truncation=True, return_tensors='pt')

token_words = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

print(token_words)