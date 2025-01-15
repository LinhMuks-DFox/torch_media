rule("libtorchmedia_dependence")
    on_load(function (target)
        -- 获取用户提供的 deps 路径
        local deps_path = target:values("deps_path") or os.projectdir() -- 默认使用项目根目录

        -- Fmt 配置
        local fmt_path = path.join(deps_path, "dependence/fmt")
        target:add("includedirs", path.join(fmt_path, "include"))
        target:add("defines", "FMT_HEADER_ONLY") -- 提前定义宏

        -- 添加 libtorch 的路径
        local libtorch_path = path.join(deps_path, "dependence/libtorch")
        target:add("includedirs", path.join(libtorch_path, "include"))
        target:add("includedirs", path.join(libtorch_path, "include/torch/csrc/api/include"))
        target:add("linkdirs", path.join(libtorch_path, "lib"))

        -- 添加 sox 的路径
        local sox_path = "/opt/homebrew/Cellar/sox/14.4.2_5"
        target:add("includedirs", path.join(sox_path, "include"))
        target:add("linkdirs", path.join(sox_path, "lib"))

        -- 设置运行时 rpath 路径
        local libtorch_abspath = path.absolute(libtorch_path)
        local sox_abs_path = path.absolute(sox_path)
        target:add("rpathdirs", path.join(libtorch_abspath, "lib"))
        target:add("rpathdirs", path.join(sox_abs_path, "lib"))

        -- 链接必要的库
        target:add("links", "torch", "torch_cpu", "c10", "sox")
    end)