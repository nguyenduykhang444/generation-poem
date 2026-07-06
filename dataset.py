import pandas as pd
from datasets import load_dataset

df = load_dataset("phamson02/vietnamese-poetry-corpus", split='train').to_pandas()
df = df.loc[df['genre'] == 'lục bát', ['content']]

df['content'] = df['content'].astype(str).str.replace(r'<\n>', '\n', regex=True)

df['content'] = df['content'].replace(
    {r'<[^>]+>': ' ', r'[^\w\s\n]': '', r'[ \t]+': ' '}, regex=True
).str.strip()

def is_luc_bat(text):
    lines = [line for line in text.split('\n') if line.strip()]
    return bool(lines) and len(lines) % 2 == 0 and all(
        len(line.split()) == (6 if i % 2 == 0 else 8) for i, line in enumerate(lines)
    )

df = df[df['content'].apply(is_luc_bat)]
print(f"Số lượng bài thơ lục bát hợp lệ: {len(df):,}")

df_train = df.sample(frac=0.8, replace=True, random_state=42)
df_test = df.drop(df_train.index).reset_index(drop=True)
df_train = df_train.reset_index(drop=True)

with open('train.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(df_train['content']))

with open('test.txt', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(df_test['content']))

print("Hoàn tất lưu file train.txt và test.txt!")