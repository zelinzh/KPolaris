include(FetchContent)

set(KPOLARIS_KOKKOS_VERSION "5.1.1" CACHE STRING "Kokkos version fetched by KPolaris")
option(KPOLARIS_FETCH_KOKKOS "Fetch and build Kokkos as part of KPolaris" ON)
option(KPOLARIS_ENABLE_CUDA "Enable the Kokkos CUDA backend" OFF)
option(KPOLARIS_ENABLE_OPENMP "Enable the Kokkos OpenMP backend" OFF)
option(KPOLARIS_ENABLE_SERIAL "Enable the Kokkos Serial backend" ON)
option(KPOLARIS_ENABLE_NATIVE_CPU_OPTIMIZATION
  "Enable native CPU Release optimizations (-march=native, -mtune=native, loop unrolling, and IPO/LTO when supported)"
  OFF)
set(KPOLARIS_CUDA_ARCH "" CACHE STRING "Kokkos NVIDIA GPU architecture, e.g. AMPERE80 or BLACKWELL120")
if(KPOLARIS_ENABLE_CUDA AND KPOLARIS_CUDA_ARCH STREQUAL "")
  message(FATAL_ERROR "Set KPOLARIS_CUDA_ARCH for the target GPU, or use scripts/kpolaris.py for automatic detection.")
endif()

if(KPOLARIS_ENABLE_NATIVE_CPU_OPTIMIZATION)
  if(KPOLARIS_ENABLE_CUDA)
    message(FATAL_ERROR "KPOLARIS_ENABLE_NATIVE_CPU_OPTIMIZATION is for CPU/OpenMP builds; do not combine it with KPOLARIS_ENABLE_CUDA.")
  endif()
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(WARNING "KPOLARIS_ENABLE_NATIVE_CPU_OPTIMIZATION is intended for Release builds; current CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
  endif()
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    $<$<COMPILE_LANGUAGE:CXX>:-mtune=native>
    $<$<COMPILE_LANGUAGE:CXX>:-funroll-loops>)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT KPOLARIS_IPO_SUPPORTED OUTPUT KPOLARIS_IPO_ERROR LANGUAGES CXX)
  if(KPOLARIS_IPO_SUPPORTED)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  else()
    message(WARNING "IPO/LTO requested by KPOLARIS_ENABLE_NATIVE_CPU_OPTIMIZATION but is not supported: ${KPOLARIS_IPO_ERROR}")
  endif()
endif()
if(KPOLARIS_FETCH_KOKKOS)
  set(Kokkos_ENABLE_SERIAL ${KPOLARIS_ENABLE_SERIAL} CACHE BOOL "KPolaris Kokkos Serial backend" FORCE)
  set(Kokkos_ENABLE_OPENMP ${KPOLARIS_ENABLE_OPENMP} CACHE BOOL "KPolaris Kokkos OpenMP backend" FORCE)
  set(Kokkos_ENABLE_CUDA ${KPOLARIS_ENABLE_CUDA} CACHE BOOL "KPolaris Kokkos CUDA backend" FORCE)
  set(Kokkos_ENABLE_AGGRESSIVE_VECTORIZATION ON CACHE BOOL "Kokkos aggressive vectorization" FORCE)

  if(KPOLARIS_ENABLE_CUDA)
    set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Enable CUDA lambdas" FORCE)
    set(Kokkos_ARCH_${KPOLARIS_CUDA_ARCH} ON CACHE BOOL "KPolaris CUDA architecture" FORCE)
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(Kokkos_ENABLE_DEBUG ON CACHE BOOL "Kokkos debug mode" FORCE)
    set(Kokkos_ENABLE_DEBUG_BOUNDS_CHECK ON CACHE BOOL "Kokkos bounds checks" FORCE)
  endif()

  FetchContent_Declare(
    kokkos
    URL https://github.com/kokkos/kokkos/releases/download/${KPOLARIS_KOKKOS_VERSION}/kokkos-${KPOLARIS_KOKKOS_VERSION}.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  FetchContent_MakeAvailable(kokkos)
  # Kokkos is a build dependency, not payload owned by KPolaris.  Excluding its
  # subdirectory from the default install prevents `cmake --install` from
  # silently copying Kokkos tools, headers, and libraries into a KPolaris
  # prefix.  Linked Kokkos targets remain part of normal KPolaris builds.
  set_property(DIRECTORY "${kokkos_SOURCE_DIR}" PROPERTY EXCLUDE_FROM_ALL TRUE)
else()
  find_package(Kokkos REQUIRED)
endif()
