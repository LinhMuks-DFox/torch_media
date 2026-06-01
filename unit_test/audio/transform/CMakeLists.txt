# unit_test/audio/transform/CMakeLists.txt
add_executable(audio_test_transform main.cpp)

target_link_libraries(audio_test_transform PRIVATE torch_media)
add_test(NAME audio_test_transform COMMAND audio_test_transform)
