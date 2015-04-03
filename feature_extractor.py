import numpy as np
#
class FeatureExtractor():

    def __init__(self):
        pass

    def fit(self, X_dict):
        pass

    def transform(self, X_dict):
        #print X_dict.keys()
        cols = [
            'magnitude_b', 
            'magnitude_r'
        ]
        X_array = np.array([[instance[col] for col in cols] for instance in X_dict])
        real_period = np.array([instance['period'] / instance['div_period']
            for instance in X_dict])
        X_array = np.concatenate((X_array.T, [real_period])).T
        return X_array
