import numpy as np
import json

from time import sleep

# importing emg acquisition utilities
from emg_comm.process_comm import EMGProcessManager
from emg_comm.myo_acquisition import MyoAcqManager 

# importing training utilities 
from pose_recognition.training import restore_element, create_mean_pipeline, RestPeriodClassifier

# defining element restauration paths
pipeline_path = "extraction_pipeline.pickle"
classifier_path = "classifier.pickle"
rest_tresh_data = "rest_tresholds.json"

# defining acquisition constants
n_clients = 4 
n_sensors = 8
buffer_depth = 50
client_check_delay = 0.15
observer_check_delay = client_check_delay/3

# defining processing constants
target_buff_len = 200

# defining the label for a rest prediction
rest_label = 4



# defining the client function
def myo_client_function(db_client):

    client_activated = False

    # defining local buffer for emg data
    emg_data_buffer = np.zeros((n_sensors, 0))

    # restoring classifier and pipeline
    extraction_pipeline = restore_element(pipeline_path)
    mov_clf = restore_element(classifier_path)

    # restoring rest classifier
    tresh_pipeline = create_mean_pipeline()
    rest_classifier = RestPeriodClassifier(rest_tresh_data)

    try : 

        while(db_client.db_model.get_incoming_data()):

            # once client is activated, delay before checking the buffer
            if client_activated : sleep(client_check_delay)
            else :
                client_activated = db_client.db_model.wait_for_start_time(db_client.client_index)

            # if data can be read from the emg data buffer
            if client_activated and db_client.db_model.get_client_read_flag(db_client.client_index):

                # adding acquired emg data to the de local buffer
                new_acq_data = np.array(json.loads(db_client.db_model.get_emg_buffer()))
                emg_data_buffer = np.hstack([emg_data_buffer, new_acq_data])

                # once buffer gets to the required size
                if emg_data_buffer.shape[1] >= target_buff_len:

                    # performing the classification
                    flat_features = np.expand_dims(emg_data_buffer.flatten(), axis=0)
                    extracted_features = extraction_pipeline.transform(flat_features)
                    prediction = mov_clf.predict(extracted_features)[0]

                    # resetting the local buffer
                    emg_data_buffer = np.zeros((n_sensors, 0))

                    # tresholding the most recent acquisition values
                    flat_curr_data = np.expand_dims(new_acq_data.flatten(), axis=0)
                    curr_acq_avgs = tresh_pipeline.transform(flat_curr_data)[0]
                    curr_is_rest = rest_classifier.is_rest_period(curr_acq_avgs)

                    # notifying the the client has read the buffer
                    db_client.db_model.lock_emg_buffer(db_client.client_index)

                    # if the latest acquisition is not a rest period
                    if not curr_is_rest or prediction == rest_label:
                        db_client.db_model.set_client_classification(db_client.client_index, prediction)
                        
    # if exception occurs, stop the acquisition
    except :
        db_client.db_model.set_incoming_data(0)
        print("Client with index : ", db_client.client_index, " has failed, acquisition process aborted.")


# defining the observer function
def myo_observer_function(db_user):

    # getting a direct reference to the redis connection
    r_connection = db_user.db_model.r_connection

    curr_classification = -1
    prev_clssification = -1
    
    try : 

        # waiting for all clients to have written a classification
        while(True):

            sleep(observer_check_delay)
            
            clients_ready = True
            for c_i in range(n_clients):
                if db_user.db_model.get_client_classification(c_i) == -1 : 
                    clients_ready = False 

            if clients_ready : break

    
        # while the data acquisition process is active
        while(db_user.db_model.get_incoming_data()):

            sleep(observer_check_delay)

            # getting current classification
            curr_classification = db_user.db_model.get_client_classification(0)

            # checking if all clients made the same classification 
            classif_consensus = True
            for c_i in range(0, n_clients):
                if not curr_classification == db_user.db_model.get_client_classification(c_i):
                    classif_consensus = False
                    break

            # writting to shared db if classification is valid
            if (not curr_classification == prev_clssification) and classif_consensus:
                r_connection.set("mov_classification", str(curr_classification))
                prev_clssification = curr_classification
                print(curr_classification)

    # if exception occurs, stop the acquisition
    except :
        db_user.db_model.set_incoming_data(0)
        print("Observer process has failed, acquisition process aborted.")



if __name__ == "__main__":

    emg_process_manager = EMGProcessManager(n_clients)
    myo_acq_manager = MyoAcqManager(emg_process_manager.data_provider, buffer_depth)

    # starting clients, observers and acquisition, then waiting to join
    emg_process_manager.launch_data_clients(myo_client_function)
    emg_process_manager.launch_data_provider(myo_acq_manager)
    emg_process_manager.launch_data_observer(myo_observer_function)
    emg_process_manager.join_data_clients()
    emg_process_manager.join_data_provider()
    emg_process_manager.join_data_observers()
