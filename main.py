import pandas as pd

df=pd.read_json('data/News_Category_Dataset_v3.json', lines=True)

print("\n First 5 rows of the data:")
print(df.head())

print("\n DataFrame info")
print(df.info())

#Check for missing values

print("\n Missing values per column:")
print(df.isnull().sum())

#Check distribution of categories

print(df['category'].value_counts())

print("\n Example headlines and descriptions:")
print(df[['headline', 'short_description', 'category']].sample(5))

from transformers import AutoTokenizer

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize_text(text):
    # This tokenizes the text and returns a dictionary with input IDs, token type IDs, and attention masks.
    return tokenizer(text, truncation=True, padding=True, max_length=128)

df['combined_text'] = df['headline'] + " " + df['short_description']

sample_text = df['combined_text'].iloc[0]
tokenized_output = tokenize_text(sample_text)

print("\nExample of tokenized text:")
print(f"Original text: {sample_text}")
print(f"Tokenized IDs: {tokenized_output['input_ids']}")
print(f"Decoded tokens: {tokenizer.convert_ids_to_tokens(tokenized_output['input_ids'])}")