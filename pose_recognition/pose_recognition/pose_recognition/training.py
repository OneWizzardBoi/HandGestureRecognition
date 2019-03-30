import numpy as np
import json

import itertools
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from pywt import wavedec

from . import raw_data

from random import shuffle

import pickle

import re


class TrainingDataGenerator():

    '''
    TrainingDataGenerator

    Pulls the training data from various training files following the structure describe in a training info file.

    All the training features are fit in a single numpy.ndarray with shape [n_samples, n_features]
    All the training labels are fit in to a single numpy.ndarray with shape [n_samples]

    The training file names must follow convention name : "(movement name)_(batch index)_(acquisition index).json"
    '''

        # training_info_f : path to the training info .json file
        # shuffle_data : (boolean), indicates if training deta should be shuffled
        # filter_funct : function to be sed for the filtering of the files by name
    def __init__(self, training_info_f, shuffle_data=True, filter_funct=None):

        self.shuffle_data = shuffle_data
        self.filter_funct = filter_funct
        self.training_info_f = training_info_f
       
        # getting training file names and details
        self.training_files = self.get_training_files()
        self.set_acquisition_dimensions()

        # getting list of all movement labels
        self.movement_labels = self.get_movement_labels()

        # generating the feature and label sets
        self.generate_training_data()


    # returns the number of acquisitions per training files 
    def set_acquisition_dimensions(self):
        # getting acquisition shape
        json_data = json.loads(open(self.training_files[0], 'r').read())
        acq_shape = np.array(json_data["emg"]["data"], dtype=np.dtype(np.int32)).shape
        # defining acquisition dimentions
        self.n_channels = acq_shape[1]
        self.acquisition_len = acq_shape[0]
        

    # returns all the de training files specified in the the training_info_f
    def get_training_files(self):

        # getting training directory paths
        try : json_file = open(self.training_info_f, 'r').read()
        except : raise(ValueError("Failed to open specified movement info .js file"))
        json_data = json.loads(json_file)["movement_info"]
        training_dirs = [movement_info["directory_path"] for movement_info in json_data]

        # getting all training files
        training_f_paths = []
        for train_dir in training_dirs:
            training_f_paths += raw_data.fetch_training_files(train_dir, self.filter_funct)

        if self.shuffle_data : shuffle(training_f_paths)
        
        return training_f_paths


    # returns a dictionary containing (key : movement name) -> (val : label)
    def get_movement_labels(self):     
 
        try:
            movement_info = json.loads(open(self.training_info_f, 'r').read())["movement_info"]
        except : raise(ValueError("Failed to open specified movement info .js file"))

        movement_labels = {}
        for mov in movement_info:
            movement_labels[mov["name"]] = mov["label"]

        return movement_labels


    # returns label for extracted movement name (movement file must respect the naming convention) 
    def get_label_for_file(self, file_name):
        try : 
            movement_name = re.search(r'(/)(?!.*/)([\w\W]+)_(\d+)_\d+.json' ,file_name).group(2)
        except : raise(Exception("File name regex failed on file : " + file_name))
        return self.movement_labels[movement_name]
        

    # generates :
        # feature set : [n_movement files, n_acquisitions * n_channels]
        # label set : [n_movement files]
    def generate_training_data(self):    

        # creating containers for all feature sets and labels
        feature_data = np.zeros((0, self.acquisition_len * self.n_channels))
        label_data = []
        
        # going through all training files
        for t_file in self.training_files : 

            # filling 1D array with acquisitions from current acquisition file
            json_data = json.loads(open(t_file, 'r').read())
            emg_data = np.swapaxes(np.array(json_data["emg"]["data"], dtype=np.dtype(np.int32)), 0,1)
            feature_data = np.vstack((feature_data, emg_data.flatten()))

            # adding the label for the current movement file
            label_data.append(int(self.get_label_for_file(t_file)))

        # setting training data attributes
        self.label_data = np.array(label_data)
        self.feature_data = feature_data
            
    # getter for label data
    def get_label_data(self):
        return self.label_data

    # getter for feature data 
    def get_feature_data(self):
        return self.feature_data

    def get_data(self):
        return self.get_feature_data(), self.get_label_data()


class Mean_Transformer(BaseEstimator, TransformerMixin):

    '''
    Mean_Transformer

    Performs feature extraction on the raw acquisition data provided
    Shape of input data : [n_samples, n_features]

    For each sample, the data is divided into (n_channels). 
    The absolute mean is computed for each of the channels.
    The means are concatenated. 
    '''
 
    # n_channels : number of chanels used for the acquisitions
    # n_avgs : number of averages to be generated per channel
    def __init__(self, n_channels=8, n_avgs=1, **kwargs):
        self.n_channels = n_channels
        self.n_avgs = n_avgs

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # making sure feature data has shape : [n_samples, n_features] 
        try: 
            if not X.ndim == 2 : 
                raise(Exception("Feature data is not properly formatted"))
        except: raise(Exception("Feature data is invalid"))        

        n_feature_sets = X.shape[0]
        n_aquisitions = X.shape[1] // self.n_channels
        tresh_len = n_aquisitions // self.n_avgs       

        # computing the absolute mean for all channels
        processed_data = np.zeros((n_feature_sets, self.n_channels * self.n_avgs))
        for f_i in range(n_feature_sets):

            for c_i in range(self.n_channels):
                                
                # computing (n_avgs averages per channel)
                channel_data = X[f_i][c_i * n_aquisitions : (c_i+1) * n_aquisitions]
                for acq_i in range(self.n_avgs):
                    processed_data[f_i][c_i + acq_i] = np.mean(np.absolute(channel_data[acq_i * tresh_len : (acq_i+1) * tresh_len]))

        return processed_data


class DWT_Transformer(BaseEstimator, TransformerMixin):

    '''
    DWT_Transformer (Discrete Wavelet Transform transformer)

    Performs feature extraction on the raw acquisition data provided.
    Shape of input data : [n_samples, n_features]
    
    For each sample, the data is first divided into (n_channels).
    The data for each channels is treated as a complex signal composed of wavelets.
    Multilevel decomposition is performed on each channel/signal at level (level).
    Coefficient statistics for each subband are compiled and concatenated.
    '''

    # level : specifies the level for wavelet decomposition
    def __init__(self, n_channels=8, level=2, **kwargs):
        
        self.level = level
        self.n_channels = n_channels

        # defining the amount of statistics extracted from coeffs of a band
        # avg and std deviation for the band
        self.n_statistics = 2

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # making sure feature data has shape : [n_samples, n_features] 
        try: 
            if not X.ndim == 2 : 
                raise(Exception("Feature data is not properly formatted"))
        except: raise(Exception("Feature data is invalid"))        

        n_feature_sets = X.shape[0]
        n_aquisitions = X.shape[1] // self.n_channels

        # going through every feature set
        samples_coeff_stats = np.zeros((n_feature_sets, self.n_channels * self.n_statistics * self.level))
        for f_i in range(n_feature_sets):

            # going through the raw acquisition data for each channel
            signals_coeff_stats = np.zeros((self.n_channels, self.level, self.n_statistics))
            for c_i in range(self.n_channels):

                # computing the coefficient bands for the signal of the current channel
                channel_raw_signal = X[f_i][c_i * n_aquisitions : (c_i+1) * n_aquisitions]
                signal_coeffs = wavedec(list(channel_raw_signal), 'db7', level=3)[1 : ]

                # extracting statistics from the subbands 
                for l_i in range(self.level):
                    mean = np.mean(np.absolute(signal_coeffs[l_i]))
                    std_dev = np.std(signal_coeffs[l_i])
                    signals_coeff_stats[c_i][l_i][ : ] = [mean, std_dev]

            # appending statistics for all feature sets
            signals_coeff_stats = signals_coeff_stats.flatten()
            samples_coeff_stats[f_i][ : ] = signals_coeff_stats

        return samples_coeff_stats


class RestPeriodClassifier():

    '''
    RestPeriodClassifier

    Allows to create the treshold values used to classify a rest period in EMG data.
    Given the predefined treshold data, performs rest period classifications

    '''

    # defining static name for treshold output file
    tresh_out_file = "rest_tresholds.json"

    # defining tolerance when comparing avg data
    tolerance = 2

    # tresh_data_in : file specifying treshold values for all emg channels 
    def __init__(self, tresh_data_in):
        self.tresh_data_in = tresh_data_in
        self.extract_tresh_data()

    # pulls tresh data from data file
    def extract_tresh_data(self):
        self.channel_tresh_vals = json.loads(open(self.tresh_data_in, 'r').read())

    # classifies provided data as a rest period or not
    def is_rest_period(self, channel_avgs):
        for avg, tresh in zip(channel_avgs, self.channel_tresh_vals):
            if avg > (tresh + self.tolerance) : return False
        return True

    # generates an array where each member represents a channel treshold value for the rest state
    # training_file : speficifies training data to be pulled to form the avgs
    @classmethod
    def create_rest_treshold(cls, training_file):

        data_generator = TrainingDataGenerator(training_file, filter_funct=raw_data.filter_in_rest)
        X, _ = data_generator.get_data()

        mean_pipeline = create_mean_pipeline()
        tresh_values = np.transpose(mean_pipeline.transform(X))

        avgs = []
        for i in range(tresh_values.shape[0]):
            avgs.append(np.mean(np.absolute(tresh_values[ : ][i])))

        # writting the treshold data to the data file
        open(cls.tresh_out_file, 'w').write(json.dumps(avgs))


# defining the averaging pipeline (does not need training)
def create_mean_pipeline(**kwargs):
    return Pipeline([
        ('mean_transformer', Mean_Transformer(**kwargs))
    ])


# defining the raw_extraction pipeline (does not need training)
# data extraction done on raw signal data, with no output normalization
def create_raw_extraction_pipeline(**kwargs):
    
    raw_extraction = FeatureUnion([('mean_transformer', Mean_Transformer(**kwargs)),
                                   ('dwt_transformer', DWT_Transformer(**kwargs))])
    return Pipeline([('raw_extraction', raw_extraction)])


# defining the feature extraction pipeline (needs training)
#   variance : variance value to be maintained from training set
#   dwt : enable Discrete Wavelet Analysis
def create_extraction_pipeline(variance=0.95, dwt=True, **kwargs): 

    # enabling frequency domain analysis
    if dwt == True : 
        raw_extraction = FeatureUnion([('mean_transformer', Mean_Transformer(**kwargs)),
                                       ('dwt_transformer', DWT_Transformer(**kwargs))])
    else : 
        raw_extraction = FeatureUnion([('mean_transformer', Mean_Transformer(**kwargs))])
    
    return Pipeline([
        ('raw_extraction', raw_extraction),
        ('pca', PCA(n_components=variance)),
        ('std_scaler', StandardScaler())
    ])


# saves object to disk
#   save_path : path to pickle file
def save_element(element, save_path=None):
    with open(save_path, 'wb') as handle:
        pickle.dump(element, handle)

# restores object from disk
#    save_path : path to pickle file
def restore_element(save_path):
    with open(save_path, 'rb') as f_handle:
        return pickle.load(f_handle)


# displays the cummulative sum for component variance
#   feature_set : array of features (shape : (n_samples, n_features))
#   variance : variance value to be maintained in data set
def display_pca_cummulative_sum(feature_set, variance=0.95):

    pca_transformer = PCA(n_components=variance)
    components = pca_transformer.fit_transform(feature_set)
    cumsum = np.cumsum(pca_transformer.explained_variance_ratio_)
    plt.plot(cumsum)
    plt.show()

    return components


# projects the spatial layout of the classes according to the two selected features
# ** colors will need to be added if the number of classes exceeds the number of colors
#   features : array of features for all the instances, shape (n_instances, n_features)
#   labels : labels associated to each feature in the feature array, shape : (n_instances)
#   d1_i : index of the first feature
#   d2_i : index of the second feature
#   label_names (optional) : movement names to map to every label 
def project_classes_2D(features, labels, d1_i, d2_i, label_names=None):

    # only keeping the wanted features
    features = np.swapaxes(features, 0, 1)
    dim_1_vals = features[d1_i][:]
    dim_2_vals = features[d2_i][:]
    
    # counting the amount of different labels
    unique_labels = np.unique(labels)
    colors = ['b', 'c', 'y', 'm', 'r']
    if(len(colors) < unique_labels.shape[0]):  
        raise(ValueError("Not enough colors defined to cover all classes."))

    # generating the scatter graph
    handles = []
    for u_label, color in zip(unique_labels, colors):

        target_indicies = np.where(np.isin(labels, [u_label]))
        features_1 = np.take(dim_1_vals, target_indicies)
        features_2 = np.take(dim_2_vals, target_indicies)

        handles.append(plt.scatter(features_1, features_2, marker='x', color=color))
        
    # generating the graph legend and title
    title_str = "features " + str(d1_i+1) + " vs " + str(d2_i+1)
    plt.title(title_str, fontdict=None, loc='center', pad=None) 
    plt.legend(handles, label_names, scatterpoints=1, loc='lower left', ncol=3, fontsize=8)

    plt.show()


# outputs a confusion matrix image in the current working directory
    # predictions : model predictions 
    # labels : actual label value for the feature set
    # label_names : dictionary with "label value" : "label name" pairs
def generate_confusion_matrix(predictions, labels, label_names=None):

    # defining file name
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    file_name = "confusion-{}.png".format(now)

    # creating the confusion matrix
    cnf_matrix = confusion_matrix(labels, predictions)

    # configuration of the display
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.colorbar()

    # putting the class names along the axies
    if label_names is not None:
        label_names = sorted(label_names, key=label_names.get)
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=90)
        plt.yticks(tick_marks, label_names)

    # placing a count on every matrix slot
    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    # saving the confusion matrix
    plt.savefig(file_name, bbox_inches='tight')