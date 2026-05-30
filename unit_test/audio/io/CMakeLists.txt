add_executable(audio_test_io main.cpp)
target_link_libraries(audio_test_io PRIVATE torch_media)
add_test(NAME audio_test_io COMMAND audio_test_io)
