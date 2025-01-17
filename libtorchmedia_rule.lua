-- for develop purpose
rule("libtorchmedia_dependence")
    on_load(function (target)

        local deps_path = target:values("deps_path") or os.projectdir()
        local fmt_path = path.join(deps_path, "dependence/fmt")
        target:add("includedirs", path.join(fmt_path, "include"))
        target:add("defines", "FMT_HEADER_ONLY") 

        local libtorch_path = path.join(deps_path, "dependence/libtorch")
        target:add("includedirs", path.join(libtorch_path, "include"))
        target:add("includedirs", path.join(libtorch_path, "include/torch/csrc/api/include"))
        target:add("linkdirs", path.join(libtorch_path, "lib"))

        local sox_path = "/opt/homebrew/Cellar/sox/14.4.2_5"
        target:add("includedirs", path.join(sox_path, "include"))
        target:add("linkdirs", path.join(sox_path, "lib"))

        local libtorch_abspath = path.absolute(libtorch_path)
        local sox_abs_path = path.absolute(sox_path)
        target:add("rpathdirs", path.join(libtorch_abspath, "lib"))
        target:add("rpathdirs", path.join(sox_abs_path, "lib"))

        target:add("links", "torch", "torch_cpu", "c10", "sox")
    end)