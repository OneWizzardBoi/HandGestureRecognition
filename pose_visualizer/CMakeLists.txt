CMAKE_MINIMUM_REQUIRED(VERSION 2.8.6)

FIND_PACKAGE(LibHand REQUIRED
  PATHS "/home/david/Documents/Code_Libraries/libhand/dist"
  NO_MODULE
  NO_DEFAULT_PATH)

INCLUDE_DIRECTORIES(${LibHand_INCLUDE_DIRS})

ADD_EXECUTABLE(hand_display main.cpp pose_display.cpp)
TARGET_LINK_LIBRARIES(hand_display ${LibHand_LIBRARIES} ${OpenCV_LIBS} libcpp_redis.a libtacopie.a)