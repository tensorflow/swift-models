add_library(StbImage STATIC
  stb_image_write.c
  stb_image.c)
target_include_directories(StbImage PUBLIC
  include)


install(TARGETS StbImage
  ARCHIVE DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  LIBRARY DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>
  RUNTIME DESTINATION bin)
get_swift_host_arch(swift_arch)
install(FILES StbImage/module.modulemap
  DESTINATION lib/swift/$<LOWER_CASE:${CMAKE_SYSTEM_NAME}>/StbImage)
