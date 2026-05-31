add_executable(plot_test main.cpp)
target_link_libraries(plot_test PRIVATE torch_media_plot)
add_test(NAME plot_test COMMAND plot_test)
