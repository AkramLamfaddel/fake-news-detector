import re
import pandas as pd 
def clean_text(text):
    #enleve les miniscule
    text=text.lower()
    #enleve les non alphanumeric
    text=re.sub(r"\W",' ',text)
    #enleve les plusieur occurence d'espace
    text=re.sub(r"\s+",' ',text)
    return text
    
data=pd.read_csv('data/news.csv')
print("fin lecture fichier")

print(data.head())


data['text']=data['text'].fillna('').apply(clean_text)

print(data.head())

print(data.isnull().sum())

data.to_csv('data/news.csv',index=False)

    


