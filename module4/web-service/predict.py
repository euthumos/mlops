import pickle
import pandas as pd
from flask import Flask, request, jsonify

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']

def preprocess(df):
    # parse incoming ISO datetime strings into real timestamps
    df['tpep_pickup_datetime']  = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    # duration in minutes
    df['duration'] = (
        df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    ).dt.total_seconds() / 60

    # filter
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    # cast categoricals
    df[categorical] = (
        df[categorical]
          .fillna(-1)
          .astype('int')
          .astype('str')
    )

    return df

def predict(df):
    dicts  = df[categorical].to_dict(orient='records')
    X_val  = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred

app = Flask('duration-prediction')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    # 1. Read payload as list-of-dicts → DataFrame
    rides = request.get_json()
    df    = pd.DataFrame(rides)

    # 2. Preprocess & predict
    df_clean = preprocess(df)
    preds    = predict(df_clean)

    # 3. Return the mean of the predictions
    mean_duration = float(preds.mean())
    return jsonify(duration=mean_duration)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)