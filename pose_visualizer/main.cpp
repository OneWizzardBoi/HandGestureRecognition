#include <unistd.h>
# include <string>

#include <cpp_redis/cpp_redis>

# include "opencv2/opencv.hpp"
# include "opencv2/highgui/highgui.hpp"

# include "file_dialog.h"
# include "hand_pose.h"
# include "hand_renderer.h"
# include "scene_spec.h"

#include "pose_display.h"

using namespace libhand;
using namespace std;
using namespace cpp_redis;

// defining the window dimensions
int w_width = 1000;
int w_height = 1000;

// defining delay params 
unsigned int usecs = 20000;
    
int main(int argc, char **argv) {

    // defining dialog box to fetch file paths
    FileDialog dialog;

    // Loading hand movement information
    dialog.SetTitle("Please select a movement information file");
    std::map<int, MovementInfo> mov_info = load_movement_data(dialog.Open());
    
    // Loading a scene spec from file on system
    dialog.SetTitle("Please select a scene spec file");
    SceneSpec scene_spec(dialog.Open());

    // Setup the hand renderer and the pose object
    HandRenderer hand_renderer;
    hand_renderer.Setup(w_width, w_height);
    hand_renderer.LoadScene(scene_spec);
    FullHandPose hand_pose(scene_spec.num_bones());

    // setting a windows and matrix for display
    std::string win_name = "Hand";
    cv::namedWindow(win_name);    
    cv::Mat mov_matrix = hand_renderer.pixel_buffer_cv();
    
    // connecting to the redis server
    int connection_status = 1;
    cpp_redis::client client;
    client.connect("127.0.0.1", 6379, [&connection_status](const std::string& host, std::size_t port, cpp_redis::client::connect_state status) {
        if (status == cpp_redis::client::connect_state::dropped) {
            std::cout << "client disconnected from " << host << ":" << port << std::endl;
            connection_status = 0;
        }
    });

    // vars for handling classification
    int curr_classification = -1, prev_classification = -1;
    
    // while connected to the redis server
    while(connection_status){

        // checking db for latest classification
        curr_classification = get_lastest_classification(client);
        
        // handling a faied request
        if(curr_classification == -1){
            std::cout << "Communication with redis failed, program aborted \n";
            connection_status = 0;
        }

        // updating the current movement classification
        if(curr_classification != prev_classification){

            // rendering new pose and updating the display
            hand_pose.Load(mov_info[curr_classification].pose_path(), scene_spec);
            hand_renderer.SetHandPose(hand_pose, true);
            hand_renderer.RenderHand();
            mov_matrix = hand_renderer.pixel_buffer_cv();
            
            std::cout << "classification : " << curr_classification << "\n";

            // updating the previous classification
            prev_classification = curr_classification;

        }

        // refreshing the image display
        cv::imshow(win_name, mov_matrix);
        cv::waitKey(5);
        usleep(usecs);

    }

    return 0;
}