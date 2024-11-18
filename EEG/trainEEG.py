import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Load data
file_path = 'C:/Users/user/Desktop/Complete_Blink_Data.csv'
df = pd.read_csv(file_path)

print(df)


X=df[['Channel_0']]
y=df['Blink']


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Split data into train and test sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# Initialize and train the classifier
clf=RandomForestClassifier()
clf.fit(X_train,y_train)

# Make predictions and evaluate
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)

print(acc)