
# News Summary and Headlines Generation

After giving a text as an input the News Summary and News Headlines will be predicted.


## Path

The Path for the files are:

`cd/home/ubuntu/work/from_hrithik/T5NewsSummary_Headlines/`



## Conda Activate

Activate the conda environment

```bash
  conda activate hrithik_T5Model
```


## Training the model and Prediction

In this folder we would
have two more folders, viz., Headlines and News_Summary. The dataset for training is inside
both the folders. In the Headlines folder, we have a python file named headlines.py Run this file
and after training is completed, it will show a list of output files directory.
The filename would be of the training and validation loss, so copy the filename with the least loss and give it as input.
After that, the text will be asked for input and the headline will be predicted. The same procedure
as above is applicable for the news_summarization.py file which is in the News_Summary
folder.