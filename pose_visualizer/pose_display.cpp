#include "pose_display.h"


// default constructor definition
MovementInfo::MovementInfo(){
    m_label = -1;
    m_name = "";
    m_pose_path = "";
}

// prints out the content of the movement info object
void MovementInfo::display_movement_info() const{

    std::cout << "\nMovement info\n";
    std::cout << "name : " << m_name << std::endl;
    std::cout << "pose_path : " << m_pose_path << std::endl;
    std::cout << "label : " << m_label << std::endl;

}

// getters and seters
std::string MovementInfo::pose_path() const {
    return m_pose_path; 
}


// loads the movement information from the provided json file into a (label, definition) map
std::map<int, MovementInfo> load_movement_data(std::string mov_info_path){

    std::map<int, MovementInfo> movement_info_map;
   
    // loading info file contents
    boost::property_tree::ptree root_node;
    boost::property_tree::read_json(mov_info_path, root_node);
    
    // loading each movement definition into map
    for(auto mov_inf_it : root_node.get_child("movement_info")){
       
        std::string name = mov_inf_it.second.get<std::string>("name").data();
        std::string pose_path = mov_inf_it.second.get<std::string>("pose_path").data();
        
        std::string s_label = mov_inf_it.second.get<std::string>("label").data();
        int label = std::stoi(s_label); 

        movement_info_map[label] = MovementInfo(name, pose_path, label);

    }

    return movement_info_map;
}


// fetches the latest classification from the shared db
int get_lastest_classification(cpp_redis::client& client){

    std::string target_key = "mov_classification";
    int classification = 0;

    // sending get request
    client.get(target_key, [&classification](cpp_redis::reply& reply) {
          if (reply.is_string()){
              classification = std::stoi(reply.as_string());
          } else {
              classification = -1;
          }    
    });

    // synchronous commit, no timeout
    client.sync_commit();

    return classification; 

}
