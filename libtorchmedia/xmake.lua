target("torch_media")
    set_kind("headeronly")
    add_headerfiles("include/(**.hpp)")
    add_includedirs("include", {public = true})
    set_basename("torch_media")
    set_license("MIT")
    on_load(function (target)
        if has_package("fmt") then
            target:add("packages", "fmt")
        end
        if has_package("libtorch") then
            target:add("packages", "libtorch")
        end
        if has_package("sox") then
            target:add("packages", "sox")
        end
    end)
