add_executable(GPT2Inference
  main.swift)
target_link_libraries(GPT2Inference PRIVATE
  TextModels)

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  add_executable(GPT2InferenceUI
    UI/Windows/main.swift)
  target_link_libraries(GPT2InferenceUI PRIVATE
    TextModels
    Foundation
    SwiftWin32
    User32)
  add_custom_command(TARGET GPT2InferenceUI POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/UI/Windows/GPT2InferenceUI.exe.manifest $<TARGET_FILE_DIR:GPT2InferenceUI>
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/UI/Windows/GPT2InferenceUI.exe.manifest )
elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
  add_executable(GPT2InferenceUI
    UI/macOS/main.swift)
  set_target_properties(GPT2InferenceUI PROPERTIES
    BUILD_RPATH @executable_path
    INSTALL_RPATH @executable_path)
  target_link_libraries(GPT2InferenceUI PRIVATE
    TextModels)
  add_custom_command(TARGET GPT2InferenceUI POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/UI/macOS/Info.plist $<TARGET_FILE_DIR:GPT2InferenceUI>
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/UI/macOS/Info.plist)
endif()
