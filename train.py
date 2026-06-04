import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


ma= pd.read_csv(r"C:\\Users\\Hiren\\Desktop\\mobile_addiction\\data\\mobile_addiction.csv")
ma.isnull().sum()

from sklearn.preprocessing import LabelEncoder
encoders = {}

for col in ma.columns:

    le = LabelEncoder()

    ma[col] = le.fit_transform(ma[col])

    encoders[col] = le
print(ma)


x= ma.drop(columns=['check phone time','addiction level'])
y= ma['addiction level']

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model=RandomForestClassifier(n_estimators=200)
model.fit(x_train,y_train)

pre=model.predict(x_test)
print(pre)
acc= accuracy_score(y_test,pre)
print("Accuracy:",acc)
fi = model.feature_importances_
print(fi)

import matplotlib.pyplot as plt
plt.barh(x.columns,fi)
plt.show()

import joblib
joblib.dump(
    model,
    "C:\\Users\\Hiren\\Desktop\\mobile_addiction\\model\\model.pkl"
)

joblib.dump(
    encoders,
    "C:\\Users\\Hiren\\Desktop\\mobile_addiction\\model\\encoders.pkl"
)

print("Saved Successfully")
