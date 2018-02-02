import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

def nans_to_med(df):
    '''
    Input: dataframe
    Output: dataframe with NaNs filled with median
    Features list includes all final features whether or not they have NaNs.
    '''
    features = ['gts',
                'num_order',
                'user_age',
                'num_payouts',
                'user_type',
                'body_length',
                'user_created',
                'name_length',
                'org_facebook']

    for col in features:
        med_val = df[col].median()
        df[col].fillna(value=med_val,inplace=True)
    return None

def load_clean_data(filename):
    df = pd.read_json(filename)
    df['fraud'] = df['acct_type'].apply(lambda x: 1 if 'fraud' in x.lower() else 0)
    nans_to_med(df)
    return df

def train_model(df):
    features = ['gts',
                'num_order',
                'user_age',
                'num_payouts',
                'user_type',
                'body_length',
                'user_created',
                'name_length',
                'org_facebook']

    X_df = df[features]
    y_df = df['fraud']

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=42)

    rfc = RandomForestClassifier(bootstrap=False,
                                 class_weight=None,
                                 criterion='gini',
                                 max_depth=20,
                                 max_features='sqrt',
                                 max_leaf_nodes=None,
                                 min_impurity_split=1e-07,
                                 min_samples_leaf=1,
                                 min_samples_split=4,
                                 min_weight_fraction_leaf=0.0,
                                 n_estimators=200,
                                 n_jobs=1,
                                 oob_score=False,
                                 random_state=1,
                                 verbose=0,
                                 warm_start=False)

    rf_model = rfc.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)

    return y_test, y_pred

def print_scores(y_test, predictions):
    print('Accuracy:', accuracy_score(y_test, predictions))
    print('Precision:', precision_score(y_test, predictions))
    print('Recall:', recall_score(y_test, predictions))
    return None

if __name__ == "__main__":
    filename = '../data/data.json'
    df = load_clean_data(filename)
    y_test, y_pred = train_model(df)
    print_scores(y_test, y_pred)













#
