import numpy as np
import itertools
import re

# importing training utilities 
from pose_recognition.training import TrainingDataGenerator, create_extraction_pipeline, create_mean_pipeline, create_raw_extraction_pipeline,\
                                      project_classes_2D, display_pca_cummulative_sum

# importing model utilities
from sklearn import svm 

# defining training file filter function
def batch_index_filter_1(file_path):
    target_batches = [1, 2, 3]
    f_name_regex = r'(/)(?!.*/)([\w\W]+)_(\d+)_\d+.json'
    f_batch_num = int(re.search(f_name_regex, file_path).group(3))
    if f_batch_num in target_batches : return True
    else : return False

# generating the training and testing data
data_generator = TrainingDataGenerator("Training_data.json", filter_funct=batch_index_filter_1)
X, y = data_generator.get_data()

# preprocesing the features (feature extraction)
extraction_pipeline = create_extraction_pipeline(variance=0.99, n_avgs=2)
extracted_features = extraction_pipeline.fit_transform(X)

######################################################################### 2D projection of features (2 features at a time)

# defining movement label names
label_names = ["index_extension", "middle_finger_extension", \
               "ring_finger_extension", "pinky_extension", "rest"]

# generating all possile feature compinations (in groups of 2)
component_indicies = list(range(extracted_features.shape[1]))
layouts = [element for element in itertools.permutations(component_indicies, r=2)]
filtered_layouts = [] 
sum_product_list = []
for layout in layouts:
    sum_product = (layout[0] + layout[1], layout[0] * layout[1])
    if not sum_product in sum_product_list: 
        filtered_layouts.append(layout)
        sum_product_list.append(sum_product) 

for layout in filtered_layouts: 
    project_classes_2D(extracted_features, y, layout[0], layout[1], label_names=label_names)

######################################################################## plotting numper of principal components vs variance
'''
test_pipeline = create_raw_extraction_pipeline(n_avgs=2)
impulse_features = test_pipeline.transform(X)
display_pca_cummulative_sum(impulse_features, 0.99)
'''