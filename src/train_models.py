import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier
import joblib, os

df = pd.read_csv('data/transformer_data_large.csv')
print('Data Shape:', df.shape)
print(df.head())

X = df.drop(['health_score','needs_maintenance'], axis=1)
y = df['needs_maintenance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=100)
model.fit(X_train, y_train)

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f'ACCURACY: {acc*100:.2f}%')
print(classification_report(y_test, pred))

os.makedirs('models', exist_ok=True)
model.save_model('models/catboost_model.cbm')
print('Model Saved to models/catboost_model.cbm')
