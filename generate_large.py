import pandas as pd
import numpy as np
np.random.seed(42)
n=100000
df = pd.DataFrame({
    'load': np.random.normal(75, 15, n),
    'oil_temp': np.random.normal(65, 10, n),
    'winding_temp': np.random.normal(70, 12, n),
    'vibration': np.random.normal(2.5, 1.0, n),
    'dissolved_gas': np.random.normal(150, 60, n),
    'age_years': np.random.randint(1, 30, n),
    'humidity': np.random.normal(60, 15, n),
})
df['health_score'] = 100 - (df['oil_temp']-60)*0.5 - (df['vibration']-2)*8 - (df['dissolved_gas']-100)*0.05 - df['age_years']*0.4 + np.random.normal(0,5,n)
df['health_score'] = df['health_score'].clip(10, 100)
df['needs_maintenance'] = (df['health_score'] < 60).astype(int)
df.to_csv('data/transformer_data_large.csv', index=False)
print('Rows:', len(df))
