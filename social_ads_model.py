import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler


df=pd.read_csv('Social_Network_Ads.csv')
df.drop(['User ID'],axis=1,inplace=True)
lr=LabelEncoder()
df.iloc[:,0] = lr.fit_transform(df.iloc[:,0])
df_majority=df[df['Purchased']==0]
df_minority=df[df['Purchased']==1]
df_minority_upsampled=resample(df_minority,replace=True,n_samples=257,random_state=42)
df_upsampled = pd.concat([df_minority_upsampled,df_majority])
x=df_upsampled.drop('Purchased',axis=1)
y=df_upsampled['Purchased']
sc=StandardScaler()
x_scaled = sc.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.25,random_state=42)
def training_model():
    model = DecisionTreeClassifier()
    trained_model = model.fit(x_train,y_train)
    return trained_model




