import pandas as pd
import numpy as np
import cPickle
from sklearn.ensemble import RandomForestClassifier

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

def build_model(df):
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

    return rfc.fit(X_df, y_df)

def pickle_me(model):
    filename = 'fraud_detect_model.pkl'
    with open(filename, 'wb') as whatever:
        cPickle.dump(model, whatever)
    return None 

if __name__ == "__main__":
    filename = '../data/data.json'
    df = load_clean_data(filename)
    model = build_model(df)
    pickle_me(model)












#
