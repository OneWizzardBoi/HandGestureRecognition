import numpy as np
import itertools

# importing training utilities 
from pose_recognition.training import TrainingDataGenerator, create_extraction_pipeline,\
                                      RestPeriodClassifier, save_element

# importing model utilities
from sklearn import svm

pipeline_path = "extraction_pipeline.pickle"
classifier_path = "classifier.pickle"
training_data_file = "Training_data.json"

# pulling and proccessing the training data
data_generator = TrainingDataGenerator(training_data_file)
X, y = data_generator.get_data()
extraction_pipeline = create_extraction_pipeline(variance=0.99, n_avgs=2)
extracted_features = extraction_pipeline.fit_transform(X)

# defining the classifier with optimized params
poly_clf = svm.SVC(kernel="poly", degree=1, C=10)
poly_clf.fit(extracted_features, y)
save_element(extraction_pipeline, pipeline_path)
save_element(poly_clf, classifier_path)

# genarating rest treshold data (for output validation)
RestPeriodClassifier.create_rest_treshold(training_data_file)