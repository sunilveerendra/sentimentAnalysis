#to download vader for the very first time
# import nltk
# nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
class Analysis:
    sia = SentimentIntensityAnalyzer()

    #add compound score to data set
    def getSentiment(self,df):

        df['scores'] = df['review'].apply(lambda review: self.sia.polarity_scores(review))

        df['compound']  = df['scores'].apply(lambda score_dict: score_dict['compound'])

        df['comp_score'] = df['compound'].apply(lambda c: 'pos' if c >=0 else 'neg')
        return df

    def getMetrics(self,df):

        score = accuracy_score(df['label'],df['comp_score'])

        report = classification_report(df['label'],df['comp_score'])

        matrix = confusion_matrix(df['label'],df['comp_score'])

        return score,report,matrix

