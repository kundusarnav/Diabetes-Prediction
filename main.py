import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
col_names=['pregnant','glucose','bp','skin','insulin','bmi','pedigree','age','label']
df=pd.read_csv('diabetes.csv',header=None,names=col_names)
df.head()
df.describe()
features_cols=['pregnant','insulin','bmi','glucose','bp','pedigree']
x=df[features_cols][1:]
y=df.label[1:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)
clf=DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf=clf.fit(x_train,y_train)
y_pred=clf.predict(x_test)
print("Accuracy is ",metrics.accuracy_score(y_test,y_pred))
y_pred[0:]
