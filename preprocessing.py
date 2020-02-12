from copy import deepcopy

import numpy
from sklearn.preprocessing import LabelEncoder


def get_features_and_labels(data, feature_fields, label=None):
    y = None

    feature_fields_with_label = deepcopy(feature_fields)
    if label:
        feature_fields_with_label.append(label)
    # features = data[feature_fields_with_label].fillna(data[feature_fields_with_label].mean())
    features = data[feature_fields_with_label]



    if label:
        y = features[label].values
        encoder = LabelEncoder()
        encoder.fit(y)
        print("Encoder classes", encoder.classes_)
        numpy.save('classes.npy', encoder.classes_)

    return features, y
