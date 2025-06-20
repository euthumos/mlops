#!/usr/bin/env python
# coding: utf-8


# In[ ]:


import pickle
import pandas as pd


# In[ ]:


import pyarrow as pa
print("pandas:", pd.__version__)
print("pyarrow:", pa.__version__)


# In[ ]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[ ]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[ ]:


df = read_data('../datasets/yellow_tripdata_2023-03.parquet')


# In[ ]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[ ]:


print('Q1: ', y_pred.std())


# In[ ]:


year = df['tpep_pickup_datetime'].dt.year.iloc[0]
month = df['tpep_pickup_datetime'].dt.month.iloc[0]

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype(str)


# In[ ]:


print(df.head())


# In[ ]:


df_result = df[['duration',	'ride_id']]

output_file = 'results.parquet'

df_result.to_parquet(
    output_file,
    engine='pyarrow',
    compression=None,
    index=False
)


# In[ ]:


df_april = read_data('../datasets/yellow_tripdata_2023-04.parquet')

dicts = df_april[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[ ]:


print('Q5: ',y_pred.mean())


# In[ ]:





# In[ ]:




