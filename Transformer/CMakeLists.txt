add_executable(TransformerDemo
  main.swift)
target_link_libraries(TransformerDemo PRIVATE
  TextModels)

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  add_executable(TransformerUI
    UI/Windows/main.swift)
  target_link_libraries(TransformerUI PRIVATE
    TextModels
    Foundation
    SwiftWin32
    User32)
  add_custom_command(TARGET TransformerUI POST_BUILD
    COMMAND
      ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/UI/Windows/TransformerUI.exe.manifest $<TARGET_FILE_DIR:TransformerUI>
    DEPENDS
      ${CMAKE_CURRENT_SOURCE_DIR}/UI/Windows/TransformerUI.exe.manifest )
endif()
