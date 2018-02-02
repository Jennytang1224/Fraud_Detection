import cPickle
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    new_data = [  0.00000000e+00,   0.00000000e+00,   3.60000000e+01,
         0.00000000e+00,   1.00000000e+00,   3.85200000e+03,
         1.25961395e+09,   6.00000000e+01,   0.00000000e+00]
    with open('fraud_detect_model.pkl', 'rb') as fid:
        rf_loaded = cPickle.load(fid)
    y_pred = rf_loaded.predict(new_data)
    print(y_pred)
