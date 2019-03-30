import numpy as np
import json 

from time import sleep
import sys

from multiprocessing import Process

# importing myo bracelet utilities
from . import myo_raw

# importing interprocess utilities
from . import process_comm



class BluetoothFailedConnection(Exception):

    def __init__(self, value=None):
        if value is None : 
            value = "Bluetooth connection with myo failed."
        else : self.value = value

    def __str__(self):
        return repr(self.value)



class MyoAcqManager(process_comm.AcqManager):

    '''
    MyoAcqManager
    Manages the communication with the myo armband.
    Ensures that the acquisition data gets written to the shared DB.
    '''

    n_sensors = 8

    # data_provider : EMG_DB_User instance
    # buff_depth : number of acquisitions per sensor in th buffer
    # n_sensors : number of sensors we want to acquire from
    def __init__(self, data_provider, buff_depth=50):

        super(MyoAcqManager, self).__init__(data_provider)
        self.buff_depth = buff_depth
        self.emg_data_buffer = np.zeros((self.n_sensors, 0))
        self.emg_data_buffer_s = ""


    # creates a connection object with the myo armband
    def create_myo_connection(self):

        # creating a MyoRaw object
        n_tries = 20
        connection_success = False
        while(not connection_success):
            try :
                m = myo_raw.MyoRaw(sys.argv[1] if len(sys.argv) >= 2 else None)
                connection_success = True
            except :
                print("Myo dongle not found")
                n_tries -= 1
                sleep(1)
                if(n_tries == 0) : raise BluetoothFailedConnection()

        # defining the handler for acquired emg data
        def emg_acq_handler(emg_data, moving, times=[]):

            # appending acquired data to the buffer
            self.emg_data_buffer = np.c_[self.emg_data_buffer, list(emg_data)]

            # serialize buffer when it reaches the wanted depth
            if self.emg_data_buffer.shape[1] >= self.buff_depth:
                self.emg_data_buffer_s = json.dumps(self.emg_data_buffer.tolist())
                self.emg_data_buffer = np.zeros((self.n_sensors, 0))

            # if buffer is locked and serialized data is available, update the buffer in shared db
            if not self.data_provider.db_model.get_emg_buffer_available() and \
               self.emg_data_buffer_s != "" : 

                self.data_provider.db_model.set_emg_buffer(self.emg_data_buffer_s)
                self.emg_data_buffer_s = ""

                data_count = self.data_provider.db_model.get_data_count() + 1
                self.data_provider.db_model.set_data_count(data_count)

                self.data_provider.db_model.release_emg_buffer()
 
        m.add_emg_handler(emg_acq_handler)
        m.connect()

        return m        


    # enters a loop where data is constantly pulled from the myo
    # this function is ment to be ran in its own process
    def start_acquisition(self):

        # connecting to the myo armband
        try : 
            myo_conn = self.create_myo_connection()
        except Exception as e :
            self.data_provider.db_model.set_incoming_data(0)
            print("\n", e)
            print("\nMyo acquisition process terminated, error occured.")
            return

        # acquiring data from the myo armband
        try :
            while(self.data_provider.db_model.get_incoming_data()):
                myo_conn.run(1)
        except : 
            try : myo_conn.disconnect()
            except : pass
            self.data_provider.db_model.set_incoming_data(0)
            print("\nMyo acquisition process terminated, error occured.")