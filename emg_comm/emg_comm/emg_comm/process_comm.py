import redis
from multiprocessing import Process



class EMG_DB_Model:

    '''
    EMG_DB_Model
    This class takes care of all communications with the redis DB.
    '''

    def __init__(self, r_connection, n_clients):
        self.r_connection = r_connection
        self.n_clients = n_clients


    # initializes variabes in the redis
    # n_clients : number of clients accessing the db variables
    def init_redis_variables(self):

        # defining client counters
        self.r_connection.set("n_clients", str(self.n_clients))
        self.set_n_active_clients(0)

        # defining buffer availability flags
        self.set_emg_buffer_available(0)
        for _ in range(self.n_clients):
            self.r_connection.lpush("client_read_data", "0")

        # 1 = data is being received from the emgs
        self.set_incoming_data(1)

        # defining data variables
        self.set_emg_buffer("")
        self.set_data_count(0)

        # defining container for client classifications
        for _ in range(self.n_clients):
            self.r_connection.lpush("client_classifications", "-1")


    # marks the emg_buffer data as available to all clients
    def release_emg_buffer(self):

        # main availability flag
        self.set_emg_buffer_available(1)

        # availability flags for all clients
        for i in range(self.n_clients):
            self.r_connection.lset("client_read_data", i, "1")


    # marks the emg buffer as read by the client with (client_index)
    # once all clients have read the buffer, it becomes locked and gets updated
    # client_index : indentifier index for the EMG_DB_Client (int)
    def lock_emg_buffer(self, client_index):

        # if flags were already flipped
        if self.get_emg_buffer_available() == 0 : return

        # flipping flag for the specified client
        self.r_connection.lset("client_read_data", client_index, "0")
     
        # not all clients need to have read the data
        data_count = self.get_data_count()
        if data_count <= self.n_clients : client_range = data_count
        else : client_range = self.n_clients

        # checking if all clients have read the buffer data
        buff_ready = True
        for i in range(client_range):
            if self.r_connection.lindex("client_read_data", i) == "1" : 
                buff_ready = False
                break
        
        # flipping main availability flag
        if buff_ready : self.set_emg_buffer_available(0)

    # checks if client with index can start receiving data
    # client_index : indentifier index for the EMG_DB_Client (int)
    # returns : boolean, true = model can receive data
    def wait_for_start_time(self, client_index):

        if self.get_data_count() > client_index :
            n_active_clients = self.get_n_active_clients() + 1
            self.set_n_active_clients(n_active_clients)
            print("Model #", client_index, " is activated")
            return True

        return False


    # client can store his classification data in shared space
    # client_index : indentifier index for the EMG_DB_Client (int)
    # classification : classification data (int)
    def set_client_classification(self, client_index, classification):
        self.r_connection.lset("client_classifications", client_index, str(classification))

    # buffer_data : new content for buffer (string)
    def set_emg_buffer(self, buffer_data):
        self.r_connection.set("emg_buffer", buffer_data)

    # data_count : number of sets of emg data received since program start (int)
    def set_data_count(self, data_count):
        self.r_connection.set("data_count", str(data_count))

    # n_active_clients : number of active clients (int)
    def set_n_active_clients(self, n_active_clients):
        self.r_connection.set("n_active_clients", str(n_active_clients))

    # emg_buffer_available : flag indicating that the emg_buffer can be read (int)
    def set_emg_buffer_available(self, emg_buffer_available):
        self.r_connection.set("emg_buffer_available", str(emg_buffer_available))

    # incoming_data : flag indicating that emg data is being received (int)
    def set_incoming_data(self, incoming_data):
        self.r_connection.set("incoming_data", str(incoming_data))


    # client_index : indentifier index for the EMG_DB_Client (int)
    def get_client_classification(self, client_index):
        return int(self.r_connection.lindex("client_classifications", client_index))

    # returns (string)
    def get_emg_buffer(self):
        return self.r_connection.get("emg_buffer")

    def get_data_count(self):
        return int(self.r_connection.get("data_count"))

    def get_n_active_clients(self):
        return int(self.r_connection.get("n_active_clients"))

    def get_emg_buffer_available(self):
        return int(self.r_connection.get("emg_buffer_available"))

    def get_incoming_data(self):
        return int(self.r_connection.get("incoming_data"))

    # client_index : indentifier index for the EMG_DB_Client (int)
    def get_client_read_flag(self, client_index):
        return int(self.r_connection.lindex("client_read_data", client_index))



class EMG_DB_User:

    '''
    EMG_DB_User
    Allows pull and modfy data from the redis DB managed by the EMGAcquisitionManager
    '''    
    def __init__(self, db_model):
        self.db_model = db_model



class EMG_DB_Client(EMG_DB_User):

    '''
    EMG_DB_Client
    Allows pull and modfy data from the redis DB managed by the EMGAcquisitionManager.
    Clients allow for synchronized acces to the shared DB data.
    '''
    def __init__(self, db_model, client_index):
        super(EMG_DB_Client, self).__init__(db_model)
        self.client_index = client_index



class EMGProcessManager:

    '''
    EMGProcessManager
    Creates and manages connections to the shared DB.
    Creates client processes and keeps their handles.
    Creates the data provider process and keeps the handle.
    '''

    def __init__(self, n_clients):

        self.n_clients = n_clients

        # creating a connection pool for shared DB
        self.conn_pool = redis.ConnectionPool(host='localhost', port=6379, db=0, 
                                              decode_responses=True)
        
        # initializing the shared db
        self.db_model = self.create_db_model()
        self.db_model.init_redis_variables()

        # creating data clients
        self.data_client_h_list = []
        self.data_client_list = []
        for c_i in range(self.n_clients):
            self.data_client_list.append(EMG_DB_Client(self.create_db_model(), c_i))

        # creating the data provider
        self.data_provider = EMG_DB_User(self.create_db_model())
        self.data_provider_h = None

        # creating the data observer containers
        self.data_observer_list = []
        self.data_observer_h_list = []


    # creates a db model made with a connection from the connection pool
    def create_db_model(self):
        r_connection = redis.StrictRedis(connection_pool=self.conn_pool)
        return EMG_DB_Model(r_connection, self.n_clients)
    

    # Launches the provided client function in (n_clients) different processes
    # The client function has to take an instance of EMG_DB_Client as its first parameter 
    def launch_data_clients(self, client_funct):

        for c_i in range(self.n_clients):
            client_h = Process(target=client_funct, args=(self.data_client_list[c_i],))
            client_h.start()
            self.data_client_h_list.append(client_h)


    # waits for the data client processes to end
    def join_data_clients(self):
        for client_h in self.data_client_h_list:
            client_h.join()


    # Launches the "start_acquisition" method of the "AcqManager" object in a seperate process.
    # The "start_acquisition" method has to take a parameter of type "EMG_DB_User" 
    def launch_data_provider(self, acq_manager):
        self.data_provider_h = Process(target=acq_manager.start_acquisition, args=())
        self.data_provider_h.start()


    # waits for the collection processes to end
    def join_data_provider(self):
        self.data_provider_h.join()

    
    # Launches the provided observer function in its own process.
    # The provided function has to take an instance of "EMG_DB_User" as its first parameter.
    # The EMG_DB_User instance provided to the "observer_function" is created and is added to the data observer list.
    def launch_data_observer(self, observer_function):

        self.data_observer_list.append(EMG_DB_User(self.create_db_model()))
 
        observer_h = Process(target=observer_function, args=(self.data_observer_list[-1],))
        observer_h.start()
        self.data_observer_h_list.append(observer_h)


    # waits for the data observer processes to end
    def join_data_observers(self):
        for handle in self.data_observer_h_list:
            handle.join()



class AcqManager:

    '''
    Base class for acquisition managers.
    This class allows emg data to be pulled from varying emg hardware setups.
    '''

    def __init__(self, data_provider):
        self.data_provider = data_provider

    # Function to be overwritten.
    # Starts and maintains the emg data acquisition process.
    # This function will be ran in its own process.
    def start_acquisition(self):
        pass