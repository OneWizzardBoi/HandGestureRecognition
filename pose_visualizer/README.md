# pose_visualizer

This project adds a visual layer to the [pose_recognition](https://google.ca) project. More specifically, it displays a 3D hand who can reproduce the different poses being classified by the classification system.

Essentially, this code hooks up to the Redis DB used by the pose_recognition software and renders the recognized hand poses. To do so, this software relies on the [libHand](http://www.libhand.org/) library and the [OpenCV](https://opencv.org/) library for the handling of graphics. Also, note that the [cpp_redis](https://github.com/cylix/cpp_redis) library was used to interface with the Redis DB.

![hand_lib_capture](file:///home/david/Documents/GIT/HandGestureRecognition/pose_visualizer/Docs/hand_lib_capture.png)  
Screenshot of the display in action

### Requirements
* linux based operating system
* C++ 11
* cmake 
* [OpenCV](https://opencv.org/)
* [libHand](https://github.com/libhand/libhand)
* [cpp_redis](https://github.com/cylix/cpp_redis)
* [pose_recognition](https://google.ca)

### Installing and Running 
##### Installing
    # Clone the project and enter the project directory
    git clone (URL to the current repo)
    cd pose_visualizer
    # Create a build directory and move into it
    mkdir build && cd build
    # Generate the Makefile using CMake and build the executable
    cmake...
    make
##### Running
    # Launch the executable (from the build folder)
    ./hand_display


### Specifying the hand poses

Before being able to use the executable, the user has to modify the contents of the "Training_data.json" file used by the [pose_recognition](https://google.ca) code. For each specified movement in the info file, the "pose_path" field must specify an absolute path to the corresponding hand pose .yml file.

All the hand pose .yml files currently implemented are located in the "pose_visualizer/hand_poses" directory.

##### Movement info entry example 

    {
        "name" : "index_extension",
        "directory_path" : "/home/david/Documents/Synapsets/work_in_progress/Training_data_v2_processed/Index_extension",
        "pose_path" : "/home/david/Documents/GIT/pose_visualizer/hand_poses/index_finger_extended.yml",
        "label" : "0" 
    }


### Using the executable

**Note that the generated executable is not a stand-alone program, it is meant to work alongside the [pose_recognition](https://google.ca) package. The movement classification system should be running before the exec is launched.**

Once the exec is running, follow these steps : 
* Provide the path to the Training_data.json file provided in the project folder
![path_to_info](file:///home/david/Documents/GIT/HandGestureRecognition/pose_visualizer/Docs/path_to_info.png)
* Provide the path to the scene specification file (pose_visualizer/hand_model/scene_spec.yml)
![path_to_spec](file:///home/david/Documents/GIT/HandGestureRecognition/pose_visualizer/Docs/path_to_scene_spec.png)
* The executable should now be connected to the classification system.