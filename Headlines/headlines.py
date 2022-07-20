import pandas as pd
from sklearn.model_selection import train_test_split

"""# Preprocessing The Data"""

df = pd.read_csv('news_summary.csv', encoding = 'Latin-1')

df=df[["headlines","text"]]

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
            batch_size = 8, max_epochs = 5, use_gpu = True)

print(os.listdir('outputs'))

"""# Inference"""

path_ = input("Enter the least error Epoch path: ")
model.load_model("t5", "outputs/"+path_, use_gpu = True)


"""# Headlines Prediction"""

text ="""
The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
"We'll be the comeback kids, all of us," he said. "We want to get our country back."
The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
"""

preprocess_text = text.strip().replace("\n","")
prepared_Text = "summarize: " + preprocess_text
print(model.predict(prepared_Text))
