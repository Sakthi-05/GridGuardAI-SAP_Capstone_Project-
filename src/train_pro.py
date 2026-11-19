import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from catboost import CatBoostClassifier
import joblib

df = pd.read_csv('data/transformer_data_large.csv')
X = df.drop(['health_score','needs_maintenance'], axis=1)
y = df['needs_maintenance']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print('=== TRAINING 2 PROFESSIONAL MODELS ===')

# Model 1: CatBoost
cb = CatBoostClassifier(iterations=500, depth=8, learning_rate=0.05, auto_class_weights='Balanced', verbose=0)
cb.fit(X_train, y_train)
cb_pred = cb.predict(X_test)
print(f'CatBoost Accuracy: {accuracy_score(y_test, cb_pred)*100:.2f}% | AUC: {roc_auc_score(y_test, cb.predict_proba(X_test)[:,1]):.3f}')
print(classification_report(y_test, cb_pred))

# Model 2: Random Forest for Comparison
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(f'RandomForest Accuracy: {accuracy_score(y_test, rf_pred)*100:.2f}% | AUC: {roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]):.3f}')

# Save Best Model
cb.save_model('models/catboost_pro.cbm')
joblib.dump(rf, 'models/rf_pro.pkl')

# Feature Importance
imp = pd.DataFrame({'feature': X.columns, 'importance': cb.get_feature_importance()}).sort_values('importance', ascending=False)
print('=== FEATURE IMPORTANCE ===')
print(imp)
imp.to_csv('results/feature_importance.csv', index=False)
print('PRO MODEL SAVED')
