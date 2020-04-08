add_library(STBImage STATIC
  stb_image_write.c
  stb_image.c)
target_include_directories(STBImage PUBLIC
  include)
