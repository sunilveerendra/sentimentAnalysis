

import pandas as pd
from Utilities.analysis import Analysis

#object creation
analysis = Analysis()

#read csv file
df = pd.read_csv('../Test data/moviereviews.tsv',sep='\t')
print("-----Reading csv file is successful-----")
print(df.head())

#check for null values
df.dropna(inplace=True)

#remove empty strings
blanks=[]
for i,lb,rv in df.itertuples():  # iterate over the DataFrame
    if type(rv)==str:            # avoid NaN values
        if rv.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list


df.drop(blanks,inplace=True)
print("-----Null values and spaces are removed-----")

#check lables
print("-----Labels count-----")
print(df['label'].value_counts())

#do sentiment analysis

df = analysis.getSentiment(df=df)
print("-----Sentiment Analysis successful-----")
print(df.head())

#validate analysis
score,report,matrix = analysis.getMetrics(df=df)

print("-----Accuracy Score-----")
score = round(score*100)
print(score)
print("-----Classification Report-----")
print(report)
print("-----Confusion Matrix-----")
print(matrix)







