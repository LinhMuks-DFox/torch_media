# unit_test/vision/io/CMakeLists.txt
add_executable(vision_test_io main.cpp)

target_link_libraries(vision_test_io PRIVATE torch_media)
