cmake_minimum_required(VERSION 2.8)

cpu_setup("random")

function(random_add_executable exe)
  cpu_add_executable(${exe} ${ARGN})
  cpu_target_link_common_libraries(${exe})
endfunction()

cpu_add_runner(run.sh)

random_add_executable(random main.cc)

