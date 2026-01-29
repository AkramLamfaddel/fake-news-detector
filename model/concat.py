import pandas as pd

fake=pd.read_csv('data/Fake/Fake.csv')
true=pd.read_csv('data/True/True.csv')

fake['label']=0
true['label']=1

data=pd.concat([fake,true],ignore_index=True)



data['text']=data['text'].fillna('')
data['title']=data['title'].fillna('')
data['text']=data['title']+' '+data['text']
data=data[['text','label']]

print("1\n",data.head())

data=data.sample(frac=1,random_state=42).reset_index(drop=True)

print("2\n",data.head())

data.to_csv('data/news.csv',index=False)

print("Data saved to data/news.csv")