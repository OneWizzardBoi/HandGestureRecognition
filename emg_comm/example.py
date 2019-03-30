import numpy as np
import json

from time import sleep

from emg_comm.process_comm import EMGProcessManager
from emg_comm.myo_acquisition import MyoAcqManager 

n_clients = 4
n_sensors = 8
buffer_depth = 50
client_check_delay = 0.1
observer_check_delay = 0.05


# defining the client function
def client_function(db_client):

    client_activated = False

    try :

        while(db_client.db_model.get_incoming_data()):

            # once client is activated, delay before checking the buffer
            if client_activated : sleep(client_check_delay)
            else :
                client_activated = db_client.db_model.wait_for_start_time(db_client.client_index)

            # if data can be read from the emg data buffer
            if client_activated and db_client.db_model.get_client_read_flag(db_client.client_index):
        
                print("Client with index : ", db_client.client_index, " received data.")
                
                # the client with index 0, prints the contents of the buffer
                if db_client.client_index == 0:
                    emg_data = np.array(json.loads(db_client.db_model.get_emg_buffer()))
                    print(emg_data)

                # notifying the the client has read the buffer
                db_client.db_model.lock_emg_buffer(db_client.client_index)

    # if exception occurs, stop the acquisition
    except :
        db_client.db_model.set_incoming_data(0)
        print("Client with index : ", db_client.client_index, " has failed, acquisition process aborted.")


# defining the observer function
def observer_function(db_user):
    
    # defining container for client classifications
    classifcations = n_clients * [0] 

    try : 
    
        # while the data acquisition process is active
        while(db_user.db_model.get_incoming_data()):

            sleep(observer_check_delay)

            # collecting client classifcations
            for c_i in range(n_clients):
                classifcations[c_i] = db_user.db_model.get_client_classification(c_i)

    # if exception occurs, stop the acquisition
    except :
        db_user.db_model.set_incoming_data(0)
        print("Observer process has failed, acquisition process aborted.")


if __name__ == "__main__":

    emg_process_manager = EMGProcessManager(n_clients)
    myo_acq_manager = MyoAcqManager(emg_process_manager.data_provider, buffer_depth)

    # starting clients, observers and acquisition, then waiting to join
    emg_process_manager.launch_data_clients(client_function)
    emg_process_manager.launch_data_provider(myo_acq_manager)
    emg_process_manager.launch_data_observer(observer_function)
    emg_process_manager.join_data_clients()
    emg_process_manager.join_data_provider()
    emg_process_manager.join_data_observers()