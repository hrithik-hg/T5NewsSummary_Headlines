import pandas as pd
import os
from sklearn.model_selection import train_test_split

"""# Preprocessing The Data"""

df = pd.read_csv('news_summary.csv', encoding = 'Latin-1')

df=df[["text","ctext"]]

"""# T5Data with Training Data Column Name:- sorce_text & target_text"""

df.columns = ["target_text","source_text"]
df = df.dropna()
df=df[["source_text","target_text"]]

"""# T5 Data with Summarize prefix"""

df["source_text"] = "summarize: " + df["source_text"]

"""# Splitting the dataset into Training and Test dataframe"""

train_df, test_df = train_test_split(df, test_size=0.3)
train_df.shape, test_df.shape

"""# Using SimpleT5 for Model Training- Instantiate, Download Pre-trained Model"""

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type = "t5", model_name='t5-base')

"""# Model Training"""

model.train(train_df = train_df[: 5000],
            eval_df = test_df[: 100],
            source_max_token_len = 128,
            target_max_token_len = 50,
            batch_size = 8, max_epochs = 1, use_gpu = False, outputdir = "outputs")

print(os.listdir('outputs'))

"""# Inference"""

path_ = input("Enter the least error Epoch path: ")
model.load_model("t5", "outputs/"+path_, use_gpu = False)


"""# Headlines Prediction"""

text = input("Enter the text:- ")

preprocess_text = text.strip().replace("\n","")
news_summary = "summarize: " + preprocess_text
print("The Summary for the News is:- ", end="")
print(model.predict(news_summary))
