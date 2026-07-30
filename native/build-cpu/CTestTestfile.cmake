# CMake generated Testfile for 
# Source directory: C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native
# Build directory: C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/build-cpu
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if(CTEST_CONFIGURATION_TYPE MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test([=[da3_c_abi_smoke]=] "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/build-cpu/Debug/da3_c_abi_smoke.exe")
  set_tests_properties([=[da3_c_abi_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;138;add_test;C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test([=[da3_c_abi_smoke]=] "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/build-cpu/Release/da3_c_abi_smoke.exe")
  set_tests_properties([=[da3_c_abi_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;138;add_test;C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test([=[da3_c_abi_smoke]=] "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/build-cpu/MinSizeRel/da3_c_abi_smoke.exe")
  set_tests_properties([=[da3_c_abi_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;138;add_test;C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;0;")
elseif(CTEST_CONFIGURATION_TYPE MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test([=[da3_c_abi_smoke]=] "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/build-cpu/RelWithDebInfo/da3_c_abi_smoke.exe")
  set_tests_properties([=[da3_c_abi_smoke]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;138;add_test;C:/Users/Frere/Source/GitRepos/ModelZoo/depth-anything-v3/native/CMakeLists.txt;0;")
else()
  add_test([=[da3_c_abi_smoke]=] NOT_AVAILABLE)
endif()
