import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import joblib

data=pd.read_csv('data/news.csv')
#vectorisation TF-IDF
tfidf=TfidfVectorizer(max_features=5000)
X=tfidf.fit_transform(data['text'])
Y=data['label']

#divise entrainement,test
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

#entrainé le model
model=LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)

#evaluation
y_predict=model.predict(x_test)
print("accurancy:",accuracy_score(y_test,y_predict))
print(classification_report(y_test,y_predict))

joblib.dump(model,"model/model.pkl")

joblib.dump(tfidf,"model/tfidf.pkl")


