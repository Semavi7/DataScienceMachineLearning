import pickle
import pandas as pd

#load model
with open('diamond_model_complete.pkl', 'rb') as f:
    saved_data = pickle.load(f)

#let's see if we can make a prediction, if model works
#i'll comment this out
X_test_scaled = pd.read_csv("testdatascaled.csv")
model = saved_data['model']
print(model.predict(X_test_scaled))