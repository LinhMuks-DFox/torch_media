# unit_test/audio/functional/CMakeLists.txt
add_executable(audio_test_functional main.cpp)

target_link_libraries(audio_test_functional PRIVATE torch_media)
add_test(NAME audio_test_functional COMMAND audio_test_functional)
