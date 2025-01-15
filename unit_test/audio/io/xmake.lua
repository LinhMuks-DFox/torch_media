includes("../../../libtorchmedia_rule.lua", "../../../libtorchmedia/xmake.lua")
target("audio_test_io")
    set_kind("binary")
    add_files("main.cpp")
    add_rules("libtorchmedia_dependence")
    set_languages("c++17")
    add_deps("torch_media")
    set_values("deps_path", "../../../") -- 传递正确的依赖路径
