# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/eanigmal/Code/MonoVulkan/tracy/profiler/.cpm-cache/capstone/c68db851083fcffa27a786090cedcce4c4f28330")
  file(MAKE_DIRECTORY "/home/eanigmal/Code/MonoVulkan/tracy/profiler/.cpm-cache/capstone/c68db851083fcffa27a786090cedcce4c4f28330")
endif()
file(MAKE_DIRECTORY
  "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-build"
  "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-subbuild/capstone-populate-prefix"
  "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-subbuild/capstone-populate-prefix/tmp"
  "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-subbuild/capstone-populate-prefix/src/capstone-populate-stamp"
  "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-subbuild/capstone-populate-prefix/src"
  "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-subbuild/capstone-populate-prefix/src/capstone-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-subbuild/capstone-populate-prefix/src/capstone-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/eanigmal/Code/MonoVulkan/tracy/profiler/_deps/capstone-subbuild/capstone-populate-prefix/src/capstone-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
