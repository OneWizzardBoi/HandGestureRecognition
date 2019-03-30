# emg_comm

This module simplifies the inter-process communication needed for EMG signal acquisition and processing.

In terms of signal processing, the module allows the user to specify his own client functions in order to handle acquired data.

For signal acquisition, the module comes pre-equipped with a communication interface for the [myo armband](https://support.getmyo.com/hc/en-us/articles/203398347-Getting-started-with-your-Myo-armband).
Other device interfaces will be implemented in the future.

### Architecture rundown

![Architecture](Docs/Architecture%20Diagram.png)
    
A Redis DB is created and shared with all processes: acquisition, clients and observers.

Acquisition from the EMG sensors happens in the acquisition process. This process then writes the EMG data to a buffer in the shared Redis DB.

The user chooses how many parallel client processes need to run at once. He also needs to define the function which will run as a client process.
An example client function is provided in the "example.py" file. 

The client function has access to the contents of the shared Redis DB and can pull the acquired data.
As seen in the example, the client process remains synchronized with the acquisition process via shared flags.

The user can also specify observer functions which also run in their own dedicated process.  
Just like a client function, an observer function has access to the contents of the shared DB. The main difference is that observer functions don't have to be synchronized to the acquisition process.
An example observer function is provided in the "example.py" file.

### System requirements
    - linux (ubuntu)
    - Python3.6
    - pip3 (sudo apt install python3-pip)
    - redis server (sudo apt-get install redis-server)

### Installing python requirements
    - pip3 install -r requirements.txt
    
### Installing the emg_comm package
    - Copy the "emg_comm" folder next to the "exemple.py" file
    - Paste the folder in the wanted location (ex : Downloaded modules folder) 
    - From the command line, navigate to the pasted folder
    - pip3 install --user -e emg_comm