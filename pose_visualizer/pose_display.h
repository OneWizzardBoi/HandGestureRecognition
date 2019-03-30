#ifndef POSE_DISPLAY_H
#define POSE_DISPLAY_H

#include <iostream>
#include <string>
#include <map>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

#include <cpp_redis/cpp_redis>

#include "hand_pose.h"

using namespace libhand;
using namespace std;
 
class MovementInfo{

    public:
        
        MovementInfo();
        MovementInfo(std::string name, std::string pose_path, int label) : 
            m_name(name), m_pose_path(pose_path), m_label(label){};
        
        void display_movement_info() const;
        
        std::string pose_path() const;

    private:
        std::string m_name;
        std::string m_pose_path; 
        int m_label; 
};

std::map<int, MovementInfo> load_movement_data(std::string mov_info_path);
int get_lastest_classification(cpp_redis::client& client);

#endif