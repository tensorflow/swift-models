add_library(STBImage STATIC
  stb_image_write.c
  stb_image.c)
set_target_properties(STBImage PROPERTIES
  POSITION_INDEPENDENT_CODE YES)
target_include_directories(STBImage PUBLIC
  include)
