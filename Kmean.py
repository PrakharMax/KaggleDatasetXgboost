
from main import *

K_MEANS_DATASET = 'myo_ds_30l_10ol_kmeans_labels.npz'
SCORING = ['accuracy', 'f1_macro', 'f1_micro']

kdata = np.load(K_MEANS_DATASET)
kX, ky = kdata['X'], kdata['y']

assert not np.any(kX != X), "A comparison with different data is not fair."

model = XGBClassifier()  # Use XGBoost classifier

results = cross_validate(model, np.median(kX, axis=1), ky, scoring=SCORING, n_jobs=-1, cv=KFold(5, shuffle=True))
for key, value in results.items():
    print(key)
    print(value)
    print('**********')

np2img = lambda x: ((x - x.min()) / (x.max() - x.min()) * 255).astype(np.uint8)
int2gesture = {
    0: 'neutral',
    1: 'flexion',
    2: 'extension',
    7: 'fist'
}

kint2gesture = {
    2: 'neutral',
    0: 'flexion',
    1: 'extension',
    3: 'fist'
}
gesture2intensity = {
    'neutral': 20,
    'flexion': 64,
    'extension': 128,
    'fist': 255
}

DATASET = 'myo_ds_30l_10ol.npz'
K_MEANS_DATASET = 'myo_ds_30l_10ol_kmeans_labels.npz'

kdata = np.load(K_MEANS_DATASET)
kX = kdata['X']
ky = kdata['y']

data = np.load(DATASET)
X = data['X']
y = data['y']

print(np.any(kX != X))
