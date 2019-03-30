from pose_recognition.raw_data import prepare_training_data

# processing contents of the specified folder
target_dirs = ["path/to/training-data/folder", "other folder path"]
for directory in target_dirs: prepare_training_data(directory)