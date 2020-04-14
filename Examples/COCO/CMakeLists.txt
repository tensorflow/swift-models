add_executable(COCOExample
  main.swift)
target_link_libraries(COCOExample PRIVATE
  Datasets)


install(TARGETS COCOExample
  DESTINATION bin)
