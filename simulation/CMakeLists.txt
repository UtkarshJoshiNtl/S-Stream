cmake_minimum_required(VERSION 3.20)
project(s_stream LANGUAGES C)

include(FetchContent)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
    GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(raylib)

FetchContent_Declare(raygui
    GIT_REPOSITORY https://github.com/raysan5/raygui.git
    GIT_TAG 4.0
    GIT_SHALLOW TRUE)
FetchContent_GetProperties(raygui)
if(NOT raygui_POPULATED)
    FetchContent_Populate(raygui)
endif()

FetchContent_Declare(tinyfiledialogs
    GIT_REPOSITORY https://git.code.sf.net/p/tinyfiledialogs/code
    GIT_TAG 1fb8c7a2b6a015bfab328e9fd6a03a3666b8cb10)
FetchContent_GetProperties(tinyfiledialogs)
if(NOT tinyfiledialogs_POPULATED)
    FetchContent_Populate(tinyfiledialogs)
endif()

add_executable(s_stream
    src/gui_main.c
    ${raygui_SOURCE_DIR}/src/raygui.h
    ${tinyfiledialogs_SOURCE_DIR}/tinyfiledialogs.c)
target_include_directories(s_stream PRIVATE
    ${raygui_SOURCE_DIR}/src
    ${tinyfiledialogs_SOURCE_DIR})
target_link_libraries(s_stream PRIVATE raylib)
if(WIN32)
    target_link_libraries(s_stream PRIVATE comdlg32 ole32)
endif()
if(MSVC)
    target_compile_options(s_stream PRIVATE /W4)
else()
    target_compile_options(s_stream PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(s_stream PRIVATE m)
endif()
add_custom_command(TARGET s_stream POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/3d-files"
        "$<TARGET_FILE_DIR:s_stream>/assets/3d-files")
