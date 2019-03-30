import numpy as np

# importing training utilities
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.externals.joblib import parallel_backend
from pose_recognition.training import TrainingDataGenerator, create_extraction_pipeline, save_element, restore_element

# importing model utilities
from sklearn import svm

search_results_path = 'search_results.pickle'

# generating the training and testing data
data_generator = TrainingDataGenerator("Training_data.json")
X, y = data_generator.get_data()

# preprocesing the features (feature extraction)
extraction_pipeline = create_extraction_pipeline(variance=0.99, n_avgs=2)
extracted_features = extraction_pipeline.fit_transform(X)

# defining the search parameters
Cs = np.logspace(-3, 3, 10)
degrees = np.array([1, 2, 3, 4, 5, 6])
param_grid = dict(C=Cs, degree=degrees)

# defining the classifier
poly_clf = svm.SVC(kernel="poly", gamma='scale')
g_search = GridSearchCV(estimator=poly_clf, param_grid=param_grid, n_jobs=2, cv=3, verbose=10)

# performing the search and exporting the results
with parallel_backend('threading'):
    g_search.fit(extracted_features, y)

save_element(g_search.cv_results_, search_results_path)