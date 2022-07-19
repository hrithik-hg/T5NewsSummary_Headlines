from sklearn.model_selection import train_test_split
import pandas as pd

"""# Preprocessing The Data"""

df = pd.read_csv('news_summary.csv', encoding = 'Latin-1')
df.head()

df=df[["text","ctext"]]
df.head()

"""# T5Data with Training Data Column Name:- sorce_text & target_text"""

df.columns = ["target_text","source_text"]
df = df.dropna()
df=df[["source_text","target_text"]]
df.head()

"""# T5 Data with Summarize prefix"""

df["source_text"] = "summarize: " + df["source_text"]
df.head()

"""# Splitting the dataset into Training and Test dataframe"""

train_df, test_df = train_test_split(df, test_size=0.3)
train_df.shape, test_df.shape

"""# Using SimpleT5 for Model Training- Instantiate, Download Pre-trained Model

# Model Training
"""

from simplet5 import SimpleT5

model = SimpleT5()
model.from_pretrained(model_type = "t5", model_name='t5-base')

model.train(train_df = train_df[: 5000],
            eval_df = test_df[: 100],
            source_max_token_len = 512,
            target_max_token_len = 128,
            batch_size = 8, max_epochs = 5, use_gpu = True)

! (cd outputs; ls)

"""# Inference"""

model.load_model("t5", "outputs/simplet5-epoch-4-train-loss-1.0731-val-loss-1.4669", use_gpu = True)

"""# News Summary Prediction"""

text ="""
The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
"We'll be the comeback kids, all of us," he said. "We want to get our country back."
The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
"""

preprocess_text = text.strip().replace("\n","")
prepared_Text = "summarize: " + preprocess_text
model.predict(prepared_Text)
