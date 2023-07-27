import numpy as np
from sklearn.model_selection import cross_validate, KFold
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
DATASET = 'myo_ds_30l_10ol.npz'
SCORING = ['accuracy', 'f1_macro', 'f1_micro']
data = np.load(DATASET)
X, y = data['X'], data['y']
#Refer https://www.kaggle.com/datasets/dcaffo/myoemgdata/code?resource=download
# Check unique class labels and remap them if needed
unique_labels = np.unique(y)
encoder = LabelEncoder()
if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
    y = encoder.fit_transform(y)
model = XGBClassifier()
results = cross_validate(model, np.median(X, axis=1), y, scoring=SCORING, n_jobs=-1, cv=KFold(5, shuffle=True))
for key, value in results.items():
    print(key)
    print(value)
    print('**********')
