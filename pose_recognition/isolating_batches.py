import numpy as np
import itertools
import re

# importing training utilities 
from pose_recognition.training import TrainingDataGenerator, create_extraction_pipeline, generate_confusion_matrix

# importing model utilities
from sklearn.metrics import accuracy_score
from sklearn import svm

def batch_index_filter_1(file_path):
    target_batches = [1,2]
    f_name_regex = r'(/)(?!.*/)([\w\W]+)_(\d+)_\d+.json'
    f_batch_num = int(re.search(f_name_regex, file_path).group(3))
    if f_batch_num in target_batches : return True
    else : return False

def batch_index_filter_2(file_path):
    target_batches = [3]
    f_name_regex = r'(/)(?!.*/)([\w\W]+)_(\d+)_\d+.json'
    f_batch_num = int(re.search(f_name_regex, file_path).group(3))
    if f_batch_num in target_batches : return True
    else : return False

# pulling and proccessing the training data
train_data_generator = TrainingDataGenerator("Training_data.json", filter_funct=batch_index_filter_1)
test_data_generator = TrainingDataGenerator("Training_data.json", filter_funct=batch_index_filter_2)
X_train, y_train = train_data_generator.get_data()
X_test, y_test = test_data_generator.get_data()

# defining a pipeline for calibration purposes
cal_pipeline = create_extraction_pipeline(variance=0.99, n_avgs=2)
cal_pipeline.fit(X_test)

# preprocessing the pulled data
extraction_pipeline = create_extraction_pipeline(variance=0.99, n_avgs=2)
train_extracted_features = extraction_pipeline.fit_transform(X_train)

cal_mean = cal_pipeline.get_params()['std_scaler'].mean_
cur_mean = extraction_pipeline.get_params()['std_scaler'].mean_
new_mean = np.append(cal_mean, cur_mean[cal_mean.shape[0] : ])
extraction_pipeline.get_params()['std_scaler'].mean_ = new_mean

test_extracted_features = extraction_pipeline.transform(X_test)

# defining the classifier and getting predictions
poly_clf = svm.SVC(kernel="poly", degree=3, C=1000)
poly_clf.fit(train_extracted_features, y_train)
y_pred = poly_clf.predict(test_extracted_features)

# generating a confusion matrix
print("Accuracy score : ", accuracy_score(y_test, y_pred))
movement_labels = train_data_generator.get_movement_labels()
generate_confusion_matrix(y_pred, y_test, movement_labels)