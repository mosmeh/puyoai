cmake_minimum_required(VERSION 2.8)

cpu_setup("reinforcekurumi")

include_directories(/opt/OpenBLAS/include)
include_directories(/home/yut/Desktop/caffe/include)
link_directories(/opt/OpenBLAS/lib)
link_directories(/home/yut/Desktop/caffe/.build_release/lib)
add_definitions(-DCPU_ONLY)

function(reinforcekurumi_add_executable exe)
  cpu_add_executable(${exe} ${ARGN})
  cpu_target_link_common_libraries(${exe})

  cpu_target_link_libraries(${exe} caffe)
  cpu_target_link_libraries(${exe} boost_system)
endfunction()


cpu_add_runner(run.sh)
cpu_add_runner(net.prototxt)
cpu_add_runner(solver.prototxt)

reinforcekurumi_add_executable(reinforcekurumi main.cc ai.cc policy_network.cc)

