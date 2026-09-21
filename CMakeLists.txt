cmake_minimum_required(VERSION 3.25)

project(KPolaris VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(KPOLARIS_USE_FLOAT "Use single precision Real in KPolaris" OFF)
option(KPOLARIS_SPECIAL_FUNCTION_FLOAT "Evaluate selected radiation special functions in single precision" OFF)
option(KPOLARIS_GRMHD_FLUID_STATE_FLOAT "Store precomputed GRMHD fluid-state interpolation cache in single precision" OFF)
option(KPOLARIS_ENABLE_WARNINGS "Enable strict compiler warnings for KPolaris targets" OFF)
option(KPOLARIS_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer on supported CPU compilers" OFF)
option(KPOLARIS_ENABLE_COVERAGE "Enable gcov-compatible coverage instrumentation for KPolaris targets" OFF)
set(KPOLARIS_MAX_FREQUENCIES "1" CACHE STRING "Maximum number of frequencies evolved in one shared-geometry kernel")
option(KPOLARIS_BUILD_EXAMPLES "Build KPolaris examples" ON)
set(KPOLARIS_CXX_STANDARD "20" CACHE STRING "C++ standard used by KPolaris; Kokkos 5 requires 20 or newer")

set(CMAKE_CXX_STANDARD ${KPOLARIS_CXX_STANDARD})
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Record immutable build provenance without embedding a build timestamp.  Git
# metadata is detected at configure time; release/source-archive builders can
# provide explicit values when .git is unavailable.
set(KPOLARIS_SOURCE_REVISION "auto" CACHE STRING
  "Source revision embedded in binaries and outputs, or 'auto' to query Git")
set(KPOLARIS_SOURCE_DIRTY "auto" CACHE STRING
  "Source dirty state embedded in binaries and outputs: auto, ON, or OFF")
set(KPOLARIS_SOURCE_FINGERPRINT "auto" CACHE STRING
  "SHA-256 source-tree fingerprint embedded in binaries and outputs, or 'auto'")
set_property(CACHE KPOLARIS_SOURCE_DIRTY PROPERTY STRINGS auto ON OFF)

find_package(Git QUIET)
set(KPOLARIS_HAVE_GIT_METADATA FALSE)
if(Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
  set(KPOLARIS_HAVE_GIT_METADATA TRUE)
endif()

if(KPOLARIS_SOURCE_REVISION STREQUAL "auto")
  if(KPOLARIS_HAVE_GIT_METADATA)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE KPOLARIS_GIT_REVISION_RESULT
      OUTPUT_VARIABLE KPOLARIS_BUILD_SOURCE_REVISION
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(NOT KPOLARIS_HAVE_GIT_METADATA OR NOT KPOLARIS_GIT_REVISION_RESULT EQUAL 0 OR
     KPOLARIS_BUILD_SOURCE_REVISION STREQUAL "")
    set(KPOLARIS_BUILD_SOURCE_REVISION "unknown")
  endif()
else()
  set(KPOLARIS_BUILD_SOURCE_REVISION "${KPOLARIS_SOURCE_REVISION}")
endif()

string(TOLOWER "${KPOLARIS_SOURCE_DIRTY}" KPOLARIS_SOURCE_DIRTY_LOWER)
if(KPOLARIS_SOURCE_DIRTY_LOWER STREQUAL "auto")
  if(KPOLARIS_HAVE_GIT_METADATA)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE KPOLARIS_GIT_STATUS_RESULT
      OUTPUT_VARIABLE KPOLARIS_GIT_STATUS_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(NOT KPOLARIS_HAVE_GIT_METADATA OR NOT KPOLARIS_GIT_STATUS_RESULT EQUAL 0)
    set(KPOLARIS_BUILD_SOURCE_DIRTY -1)
  elseif(KPOLARIS_GIT_STATUS_OUTPUT STREQUAL "")
    set(KPOLARIS_BUILD_SOURCE_DIRTY 0)
  else()
    set(KPOLARIS_BUILD_SOURCE_DIRTY 1)
  endif()
elseif(KPOLARIS_SOURCE_DIRTY_LOWER MATCHES "^(1|on|true|yes)$")
  set(KPOLARIS_BUILD_SOURCE_DIRTY 1)
elseif(KPOLARIS_SOURCE_DIRTY_LOWER MATCHES "^(0|off|false|no)$")
  set(KPOLARIS_BUILD_SOURCE_DIRTY 0)
else()
  message(FATAL_ERROR
    "KPOLARIS_SOURCE_DIRTY must be auto, ON, or OFF; got '${KPOLARIS_SOURCE_DIRTY}'")
endif()

# A commit id plus a dirty flag does not identify a modified scientific
# executable.  Hash the committed revision, tracked diff, and the contents of
# untracked build-input files so outputs from a dirty tree remain traceable.
# Mark those files as configure dependencies so the fingerprint is refreshed
# before a subsequent build after an edit.
file(GLOB_RECURSE KPOLARIS_PROVENANCE_INPUTS
  LIST_DIRECTORIES FALSE
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/tools/*"
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*")
list(APPEND KPOLARIS_PROVENANCE_INPUTS "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
  ${KPOLARIS_PROVENANCE_INPUTS})

if(KPOLARIS_SOURCE_FINGERPRINT STREQUAL "auto")
  if(KPOLARIS_HAVE_GIT_METADATA AND NOT KPOLARIS_BUILD_SOURCE_REVISION STREQUAL "unknown")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" diff --binary --no-ext-diff HEAD --
              CMakeLists.txt cmake src tools
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE KPOLARIS_GIT_DIFF_RESULT
      OUTPUT_VARIABLE KPOLARIS_GIT_DIFF_OUTPUT
      ERROR_QUIET)
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" ls-files --others --exclude-standard --
              CMakeLists.txt cmake src tools
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      RESULT_VARIABLE KPOLARIS_GIT_UNTRACKED_RESULT
      OUTPUT_VARIABLE KPOLARIS_GIT_UNTRACKED_OUTPUT
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
  endif()
  if(KPOLARIS_HAVE_GIT_METADATA AND KPOLARIS_GIT_DIFF_RESULT EQUAL 0 AND
     KPOLARIS_GIT_UNTRACKED_RESULT EQUAL 0)
    set(KPOLARIS_FINGERPRINT_MATERIAL
      "revision=${KPOLARIS_BUILD_SOURCE_REVISION}\n${KPOLARIS_GIT_DIFF_OUTPUT}")
    if(NOT KPOLARIS_GIT_UNTRACKED_OUTPUT STREQUAL "")
      string(REPLACE "\n" ";" KPOLARIS_GIT_UNTRACKED_FILES
        "${KPOLARIS_GIT_UNTRACKED_OUTPUT}")
      list(SORT KPOLARIS_GIT_UNTRACKED_FILES)
      foreach(KPOLARIS_UNTRACKED_FILE IN LISTS KPOLARIS_GIT_UNTRACKED_FILES)
        if(NOT KPOLARIS_UNTRACKED_FILE STREQUAL "" AND
           EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${KPOLARIS_UNTRACKED_FILE}")
          file(SHA256
            "${CMAKE_CURRENT_SOURCE_DIR}/${KPOLARIS_UNTRACKED_FILE}"
            KPOLARIS_UNTRACKED_SHA256)
          string(APPEND KPOLARIS_FINGERPRINT_MATERIAL
            "\nuntracked=${KPOLARIS_UNTRACKED_FILE}:${KPOLARIS_UNTRACKED_SHA256}")
        endif()
      endforeach()
    endif()
    string(SHA256 KPOLARIS_BUILD_SOURCE_FINGERPRINT
      "${KPOLARIS_FINGERPRINT_MATERIAL}")
  else()
    # Archives have no commit graph; fingerprint the actual build inputs.
    list(SORT KPOLARIS_PROVENANCE_INPUTS)
    set(KPOLARIS_FINGERPRINT_MATERIAL "source-files-v1\n")
    foreach(KPOLARIS_INPUT IN LISTS KPOLARIS_PROVENANCE_INPUTS)
      file(RELATIVE_PATH KPOLARIS_INPUT_NAME "${CMAKE_CURRENT_SOURCE_DIR}" "${KPOLARIS_INPUT}")
      file(SHA256 "${KPOLARIS_INPUT}" KPOLARIS_INPUT_SHA256)
      string(APPEND KPOLARIS_FINGERPRINT_MATERIAL "${KPOLARIS_INPUT_NAME}:${KPOLARIS_INPUT_SHA256}\n")
    endforeach()
    string(SHA256 KPOLARIS_BUILD_SOURCE_FINGERPRINT "${KPOLARIS_FINGERPRINT_MATERIAL}")
  endif()
else()
  set(KPOLARIS_BUILD_SOURCE_FINGERPRINT "${KPOLARIS_SOURCE_FINGERPRINT}")
endif()

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/kpolaris_version.hpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/generated/common/version.hpp"
  @ONLY)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(KPolarisKokkos)

set(KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS_DEFAULT OFF)
if(KPOLARIS_ENABLE_CUDA)
  set(KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS_DEFAULT ON)
endif()
option(KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS
  "Use Kokkos RandomAccess traits for read-only GRMHD interpolation views; defaults ON for CUDA builds and OFF otherwise"
  ${KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS_DEFAULT})

set(KPOLARIS_BUILD_TESTS_DEFAULT ON)
if(KPOLARIS_ENABLE_CUDA)
  set(KPOLARIS_BUILD_TESTS_DEFAULT OFF)
endif()
option(KPOLARIS_BUILD_TESTS "Build KPolaris tests" ${KPOLARIS_BUILD_TESTS_DEFAULT})

option(KPOLARIS_BUILD_AUX_TOOLS "Build small smoke/probe/demo executables" OFF)
option(KPOLARIS_BUILD_TRACE_TOOL "Build the HDF5 trace executable" OFF)
option(KPOLARIS_BUILD_IMAGE_TOOL "Build model-specific image executables" ON)
option(KPOLARIS_IMAGE_REQUIRE_GRMHD_DERIVED_SCALARS
  "Require GRMHD image targets to precompute and interpolate derived plasma scalars; disable only for matched-validation builds that derive plasma scalars after primitive interpolation"
  ON)
set(KPOLARIS_IMAGE_MODELS "riaf" CACHE STRING "Semicolon-separated image models to build: riaf;binary_riaf;torus;iharm;kharma;athenak;bhac;hamr or all")
set(KPOLARIS_GRMHD_COORDINATES "fmks" CACHE STRING "Semicolon-separated GRMHD image coordinate backends: cartesian_ks;spherical_ks;boyer_lindquist;fmks;mks or all")
set(KPOLARIS_RIAF_COORDINATES "boyer_lindquist" CACHE STRING "Semicolon-separated RIAF image coordinate backends: boyer_lindquist or all")
set(KPOLARIS_TORUS_COORDINATES "cartesian_ks" CACHE STRING "Semicolon-separated torus image coordinate backends: cartesian_ks or all")
set(KPOLARIS_RIAF_EMISSION_TYPE "1" CACHE STRING "Compile-time RIAF thermal synchrotron fit: 1=Symphony/Pandya, 4=Dexter, 0=runtime branch")
set(KPOLARIS_GRMHD_EMISSION_TYPE "4" CACHE STRING "Compile-time GRMHD thermal synchrotron fit: 1=Symphony/Pandya, 4=Dexter, 0=runtime branch")
set(KPOLARIS_IHARM_EMISSION_TYPE "" CACHE STRING "Deprecated alias for KPOLARIS_GRMHD_EMISSION_TYPE")
if(NOT KPOLARIS_IHARM_EMISSION_TYPE STREQUAL "")
  set(KPOLARIS_GRMHD_EMISSION_TYPE "${KPOLARIS_IHARM_EMISSION_TYPE}" CACHE STRING "Compile-time GRMHD thermal synchrotron fit" FORCE)
endif()
set_property(CACHE KPOLARIS_RIAF_EMISSION_TYPE PROPERTY STRINGS 0 1 4)
set_property(CACHE KPOLARIS_GRMHD_EMISSION_TYPE PROPERTY STRINGS 0 1 4)
option(KPOLARIS_ENABLE_CSV_OUTPUT "Compile optional CSV image output support" OFF)
option(KPOLARIS_ENABLE_ANALYSIS_MODE "Compile optional physical analysis map output support" ON)
option(KPOLARIS_ENABLE_SLOW_LIGHT "Compile optional slow-light image transport support" ON)
option(KPOLARIS_ENABLE_DDC_COMPACT_CUDA "Enable experimental compact DDC GPU reconstruction" OFF)
set(KPOLARIS_DDC_CODEC_INCLUDE_DIR "" CACHE PATH "DDC public CUDA transport header directory")

# Astrophysical invariant emissivities routinely reach ~1e-45 at 230 GHz,
# below the useful dynamic range of a float main state.  A float build can be
# used for geometry/development experiments, but silently producing zero-valued
# science images would be worse than rejecting the configuration.  The
# special-function and GRMHD-cache float options retain double transport state.
if(KPOLARIS_USE_FLOAT AND
   (KPOLARIS_BUILD_IMAGE_TOOL OR KPOLARIS_BUILD_TRACE_TOOL))
  message(FATAL_ERROR
    "KPOLARIS_USE_FLOAT is not supported by the production image/trace tools: "
    "invariant synchrotron coefficients underflow in main float precision. "
    "Keep the main Real type double and use KPOLARIS_SPECIAL_FUNCTION_FLOAT "
    "or KPOLARIS_GRMHD_FLUID_STATE_FLOAT for mixed-precision studies.")
endif()

set(KPOLARIS_CUDA_FAST_COMPILE_LEVEL "0" CACHE STRING "Value passed to nvcc_wrapper/nvcc --Ofast-compile: 0, min, mid, or max")
set_property(CACHE KPOLARIS_CUDA_FAST_COMPILE_LEVEL PROPERTY STRINGS 0 min mid max)

foreach(fit_name KPOLARIS_RIAF_EMISSION_TYPE KPOLARIS_GRMHD_EMISSION_TYPE)
  if(NOT ${fit_name} STREQUAL "0" AND NOT ${fit_name} STREQUAL "1" AND NOT ${fit_name} STREQUAL "4")
    message(FATAL_ERROR "${fit_name} must be 0, 1, or 4")
  endif()
endforeach()

add_library(kpolaris INTERFACE)
target_include_directories(kpolaris INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/generated>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/kpolaris>
)
target_link_libraries(kpolaris INTERFACE Kokkos::kokkos)
target_compile_features(kpolaris INTERFACE cxx_std_20)

if(KPOLARIS_ENABLE_WARNINGS)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(kpolaris INTERFACE
      $<BUILD_INTERFACE:-Wall>
      $<BUILD_INTERFACE:-Wextra>
      $<BUILD_INTERFACE:-Wpedantic>)
  elseif(MSVC)
    target_compile_options(kpolaris INTERFACE $<BUILD_INTERFACE:/W4>)
  else()
    message(WARNING "KPOLARIS_ENABLE_WARNINGS has no configured flags for ${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

if(KPOLARIS_ENABLE_SANITIZERS)
  if(KPOLARIS_ENABLE_CUDA)
    message(FATAL_ERROR "KPOLARIS_ENABLE_SANITIZERS is a CPU-only validation option")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(kpolaris INTERFACE
      $<BUILD_INTERFACE:-fsanitize=address,undefined>
      $<BUILD_INTERFACE:-fno-omit-frame-pointer>)
    target_link_options(kpolaris INTERFACE
      $<BUILD_INTERFACE:-fsanitize=address,undefined>
      $<BUILD_INTERFACE:-fno-omit-frame-pointer>)
  else()
    message(FATAL_ERROR
      "KPOLARIS_ENABLE_SANITIZERS requires a GNU- or Clang-compatible CPU compiler")
  endif()
endif()

if(KPOLARIS_ENABLE_COVERAGE)
  if(KPOLARIS_ENABLE_CUDA)
    message(FATAL_ERROR "KPOLARIS_ENABLE_COVERAGE is a CPU-only validation option")
  elseif(KPOLARIS_ENABLE_SANITIZERS)
    message(FATAL_ERROR
      "KPOLARIS_ENABLE_COVERAGE and KPOLARIS_ENABLE_SANITIZERS must use separate validation builds")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    # BUILD_INTERFACE keeps instrumentation local to KPolaris and prevents an
    # installed header-only consumer from inheriting a libgcov dependency.
    # In particular, this avoids instrumenting the separately built Kokkos
    # dependency, so the source-archive and installation tests remain valid.
    target_compile_options(kpolaris INTERFACE $<BUILD_INTERFACE:--coverage>)
    target_link_options(kpolaris INTERFACE $<BUILD_INTERFACE:--coverage>)
  else()
    message(FATAL_ERROR
      "KPOLARIS_ENABLE_COVERAGE requires a GNU- or Clang-compatible CPU compiler")
  endif()
endif()

if(KPOLARIS_ENABLE_CUDA AND NOT KPOLARIS_CUDA_FAST_COMPILE_LEVEL STREQUAL "0")
  target_compile_options(kpolaris INTERFACE --Ofast-compile=${KPOLARIS_CUDA_FAST_COMPILE_LEVEL})
endif()

if(KPOLARIS_MAX_FREQUENCIES LESS 1)
  message(FATAL_ERROR "KPOLARIS_MAX_FREQUENCIES must be >= 1")
endif()
target_compile_definitions(kpolaris INTERFACE KPOLARIS_MAX_FREQUENCIES=${KPOLARIS_MAX_FREQUENCIES})
if(KPOLARIS_ENABLE_DDC_COMPACT_CUDA)
  if(NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR "Compact DDC reconstruction requires Kokkos CUDA")
  endif()
  foreach(header ddc_gpu_reconstruct.cuh ddc_compact_receive.hpp ddc_compact_types.hpp)
    if(NOT EXISTS "${KPOLARIS_DDC_CODEC_INCLUDE_DIR}/${header}")
      message(FATAL_ERROR "Missing DDC CUDA header: ${header}")
    endif()
  endforeach()
  target_include_directories(kpolaris INTERFACE $<BUILD_INTERFACE:${KPOLARIS_DDC_CODEC_INCLUDE_DIR}>)
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_DDC_COMPACT_CUDA=1)
endif()

if(KPOLARIS_SPECIAL_FUNCTION_FLOAT)
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_SPECIAL_FUNCTION_FLOAT=1)
else()
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_SPECIAL_FUNCTION_FLOAT=0)
endif()

if(KPOLARIS_USE_FLOAT)
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_USE_FLOAT=1)
else()
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_USE_FLOAT=0)
endif()

if(KPOLARIS_GRMHD_FLUID_STATE_FLOAT)
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_GRMHD_FLUID_STATE_FLOAT=1)
else()
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_GRMHD_FLUID_STATE_FLOAT=0)
endif()

if(KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS)
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS=1)
else()
  target_compile_definitions(kpolaris INTERFACE KPOLARIS_GRMHD_RANDOM_ACCESS_VIEWS=0)
endif()

if(KPOLARIS_BUILD_IMAGE_TOOL OR KPOLARIS_BUILD_TRACE_TOOL OR
   (KPOLARIS_BUILD_EXAMPLES AND KPOLARIS_BUILD_AUX_TOOLS))
  find_package(HDF5 REQUIRED COMPONENTS CXX)
elseif(KPOLARIS_BUILD_TESTS)
  find_package(HDF5 QUIET COMPONENTS CXX)
endif()

option(KPOLARIS_BUILD_BENCHMARKS "Build in-memory GRRT batch benchmarks" OFF)
if(KPOLARIS_BUILD_BENCHMARKS)
  add_executable(kpolaris_riaf_batch_benchmark examples/grrt_batch_benchmark.cpp)
  target_include_directories(kpolaris_riaf_batch_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(kpolaris_riaf_batch_benchmark PRIVATE KPOLARIS_RIAF_COMPILED_EMISSION_TYPE=1)
  target_link_libraries(kpolaris_riaf_batch_benchmark PRIVATE kpolaris)
  add_executable(kpolaris_athenak_batch_benchmark examples/grrt_batch_benchmark.cpp src/grmhd/athenak_loader.cpp)
  target_include_directories(kpolaris_athenak_batch_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
  target_compile_definitions(kpolaris_athenak_batch_benchmark PRIVATE KPOLARIS_BENCHMARK_ATHENAK=1 KPOLARIS_IHARM_COMPILED_EMISSION_TYPE=4)
  target_link_libraries(kpolaris_athenak_batch_benchmark PRIVATE kpolaris)
endif()

if(KPOLARIS_BUILD_TESTS)
  enable_testing()
  add_executable(kpolaris_unit_tests tests/test_core.cpp)
  target_link_libraries(kpolaris_unit_tests PRIVATE kpolaris)
  add_test(NAME kpolaris_unit_tests COMMAND kpolaris_unit_tests)
  add_executable(kpolaris_equatorial_tests tests/test_equatorial.cpp)
  target_link_libraries(kpolaris_equatorial_tests PRIVATE kpolaris)
  add_test(NAME kpolaris_equatorial_tests COMMAND kpolaris_equatorial_tests)
  add_executable(kpolaris_direct_only_tests tests/test_direct_only.cpp)
  target_link_libraries(kpolaris_direct_only_tests PRIVATE kpolaris)
  add_test(NAME kpolaris_direct_only_tests COMMAND kpolaris_direct_only_tests)
  add_executable(kpolaris_physical_response_tests tests/test_physical_response.cpp)
  target_link_libraries(kpolaris_physical_response_tests PRIVATE kpolaris)
  add_test(NAME kpolaris_physical_response_tests COMMAND kpolaris_physical_response_tests)
  add_executable(kpolaris_image_batch_tests tests/test_image_batch.cpp)
  target_link_libraries(kpolaris_image_batch_tests PRIVATE kpolaris)
  add_test(NAME kpolaris_image_batch_tests COMMAND kpolaris_image_batch_tests)
  add_executable(kpolaris_fluid_time_interpolation_tests tests/test_fluid_time_interpolation.cpp)
  target_link_libraries(kpolaris_fluid_time_interpolation_tests PRIVATE kpolaris)
  add_test(NAME kpolaris_fluid_time_interpolation_tests COMMAND kpolaris_fluid_time_interpolation_tests)
  add_executable(kpolaris_slow_light_batch_tests tests/test_slow_light_batch.cpp)
  target_link_libraries(kpolaris_slow_light_batch_tests PRIVATE kpolaris)
  target_compile_definitions(kpolaris_slow_light_batch_tests PRIVATE
    KPOLARIS_ENABLE_ANALYSIS_MODE=$<BOOL:${KPOLARIS_ENABLE_ANALYSIS_MODE}>)
  add_test(NAME kpolaris_slow_light_batch_tests COMMAND kpolaris_slow_light_batch_tests)
  add_executable(kpolaris_athenak_sampling_tests tests/test_athenak_sampling.cpp)
  target_link_libraries(kpolaris_athenak_sampling_tests PRIVATE kpolaris)
  add_test(NAME kpolaris_athenak_sampling_tests COMMAND kpolaris_athenak_sampling_tests)


  if(NOT KPOLARIS_USE_FLOAT)
    add_executable(kpolaris_binary_sks_tests tests/test_binary_sks.cpp)
    target_link_libraries(kpolaris_binary_sks_tests PRIVATE kpolaris)
    add_test(NAME kpolaris_binary_sks_tests COMMAND kpolaris_binary_sks_tests)
    add_executable(kpolaris_binary_riaf_physics_tests tests/test_binary_riaf_physics.cpp)
    target_link_libraries(kpolaris_binary_riaf_physics_tests PRIVATE kpolaris)
    add_test(NAME kpolaris_binary_riaf_physics_tests COMMAND kpolaris_binary_riaf_physics_tests)
    if(HDF5_FOUND)
      add_executable(kpolaris_binary_trajectory_hdf5_tests
        tests/test_binary_trajectory_hdf5.cpp
        src/geometry/binary_trajectory_hdf5.cpp)
      target_include_directories(kpolaris_binary_trajectory_hdf5_tests PRIVATE
        ${HDF5_INCLUDE_DIRS})
      target_link_libraries(kpolaris_binary_trajectory_hdf5_tests PRIVATE
        kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
      add_test(NAME kpolaris_binary_trajectory_hdf5_tests
        COMMAND kpolaris_binary_trajectory_hdf5_tests)
    endif()
    add_executable(kpolaris_numerical_validation tests/numerical_validation.cpp)
    target_link_libraries(kpolaris_numerical_validation PRIVATE kpolaris)
    add_test(NAME kpolaris_numerical_validation
      COMMAND kpolaris_numerical_validation)
    add_executable(kpolaris_symphony_grid_probe
      tests/kpolaris_symphony_grid_driver.cpp)
    target_link_libraries(kpolaris_symphony_grid_probe PRIVATE kpolaris)
  endif()
endif()

if(KPOLARIS_BUILD_EXAMPLES AND KPOLARIS_BUILD_AUX_TOOLS)
    add_executable(kpolaris_toy_image examples/toy_image.cpp)
    target_link_libraries(kpolaris_toy_image PRIVATE kpolaris)

    add_executable(kpolaris_double_pass_toy examples/double_pass_toy.cpp)
    target_link_libraries(kpolaris_double_pass_toy PRIVATE kpolaris)

    add_executable(kpolaris_pass_a_smoke examples/pass_a_smoke.cpp)
    target_link_libraries(kpolaris_pass_a_smoke PRIVATE kpolaris)

    add_executable(kpolaris_pass_b_smoke examples/pass_b_smoke.cpp)
    target_link_libraries(kpolaris_pass_b_smoke PRIVATE kpolaris)

    add_executable(kpolaris_riaf_coeff_probe examples/riaf_coeff_probe.cpp)
    target_link_libraries(kpolaris_riaf_coeff_probe PRIVATE kpolaris)

    add_executable(kpolaris_torus_coeff_probe examples/torus_coeff_probe.cpp)
    target_link_libraries(kpolaris_torus_coeff_probe PRIVATE kpolaris)

    add_executable(kpolaris_riaf_trace_csv examples/riaf_trace_csv.cpp)
    target_link_libraries(kpolaris_riaf_trace_csv PRIVATE kpolaris)

    add_executable(kpolaris_nonthermal_coeff_probe examples/nonthermal_coeff_probe.cpp)
    target_link_libraries(kpolaris_nonthermal_coeff_probe PRIVATE kpolaris)

    add_executable(kpolaris_kharma_state_probe examples/kharma_state_probe.cpp src/grmhd/kharma_loader.cpp)
    target_include_directories(kpolaris_kharma_state_probe PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_kharma_state_probe PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_athenak_state_probe examples/athenak_state_probe.cpp src/grmhd/athenak_loader.cpp)
    target_link_libraries(kpolaris_athenak_state_probe PRIVATE kpolaris)

    add_executable(kpolaris_bhac_state_probe examples/bhac_state_probe.cpp src/grmhd/bhac_loader.cpp)
    target_link_libraries(kpolaris_bhac_state_probe PRIVATE kpolaris)
endif()

set(KPOLARIS_INSTALL_RUNTIME_TARGETS)
if(KPOLARIS_BUILD_IMAGE_TOOL)
    set(KPOLARIS_REQUESTED_IMAGE_MODELS ${KPOLARIS_IMAGE_MODELS})
    if(KPOLARIS_REQUESTED_IMAGE_MODELS STREQUAL "all")
      set(KPOLARIS_REQUESTED_IMAGE_MODELS riaf binary_riaf torus iharm kharma athenak bhac hamr)
    endif()

    foreach(model_name IN LISTS KPOLARIS_REQUESTED_IMAGE_MODELS)
      string(TOLOWER "${model_name}" model_name_lower)
      set(enable_riaf 0)
      set(enable_binary_riaf 0)
      set(enable_torus 0)
      set(enable_iharm 0)
      set(enable_kharma 0)
      set(enable_athenak 0)
      set(enable_bhac 0)
      set(enable_hamr 0)
      set(enable_grmhd 0)
      if(model_name_lower STREQUAL "riaf")
        set(enable_riaf 1)
      elseif(model_name_lower STREQUAL "binary_riaf")
        set(enable_riaf 1)
        set(enable_binary_riaf 1)
      elseif(model_name_lower STREQUAL "torus")
        set(enable_torus 1)
      elseif(model_name_lower STREQUAL "iharm")
        set(enable_iharm 1)
        set(enable_grmhd 1)
      elseif(model_name_lower STREQUAL "kharma")
        set(enable_kharma 1)
        set(enable_grmhd 1)
      elseif(model_name_lower STREQUAL "athenak")
        set(enable_athenak 1)
        set(enable_grmhd 1)
      elseif(model_name_lower STREQUAL "bhac")
        set(enable_bhac 1)
        set(enable_grmhd 1)
      elseif(model_name_lower STREQUAL "hamr")
        set(enable_hamr 1)
        set(enable_grmhd 1)
      else()
        message(FATAL_ERROR "Unknown KPolaris image model '${model_name}'. Use riaf, binary_riaf, torus, iharm, kharma, athenak, bhac, hamr, or all.")
      endif()

      set(target_name kpolaris_model_image_${model_name_lower})
      add_executable(${target_name}
        tools/kpolaris_model_image.cpp
        tools/kpolaris_model_image_options.cpp
        tools/kpolaris_model_image_output.cpp)
      set(enable_coord_cartesian_ks 0)
      set(enable_coord_spherical_ks 0)
      set(enable_coord_boyer_lindquist 0)
      set(enable_coord_fmks 0)
      set(enable_coord_mks 0)
      set(enable_riaf_coord_boyer_lindquist 0)
      set(enable_torus_coord_cartesian_ks 0)
      if(enable_riaf)
        target_sources(${target_name} PRIVATE
          src/image/riaf.cpp)
        if(enable_binary_riaf)
          target_sources(${target_name} PRIVATE
            src/image/binary_riaf.cpp
            src/geometry/binary_trajectory_hdf5.cpp)
        endif()
        set(requested_riaf_coordinates ${KPOLARIS_RIAF_COORDINATES})
        if(requested_riaf_coordinates STREQUAL "all")
          set(requested_riaf_coordinates boyer_lindquist)
        endif()
        foreach(coord_name IN LISTS requested_riaf_coordinates)
          string(TOLOWER "${coord_name}" coord_name_lower)
          if(coord_name_lower STREQUAL "boyer_lindquist" OR coord_name_lower STREQUAL "bl")
            set(enable_riaf_coord_boyer_lindquist 1)
            target_sources(${target_name} PRIVATE src/image/backends/riaf_bl.cpp)
          else()
            message(FATAL_ERROR "Unknown KPOLARIS_RIAF_COORDINATES entry '${coord_name}'. Use boyer_lindquist or all.")
          endif()
        endforeach()
      endif()
      if(enable_torus)
        target_sources(${target_name} PRIVATE
          src/image/torus.cpp)
        set(requested_torus_coordinates ${KPOLARIS_TORUS_COORDINATES})
        if(requested_torus_coordinates STREQUAL "all")
          set(requested_torus_coordinates cartesian_ks)
        endif()
        foreach(coord_name IN LISTS requested_torus_coordinates)
          string(TOLOWER "${coord_name}" coord_name_lower)
          if(coord_name_lower STREQUAL "cartesian_ks" OR coord_name_lower STREQUAL "cartesian" OR coord_name_lower STREQUAL "ks_cartesian")
            set(enable_torus_coord_cartesian_ks 1)
            target_sources(${target_name} PRIVATE src/image/backends/torus_cartesian_ks.cpp)
          else()
            message(FATAL_ERROR "Unknown KPOLARIS_TORUS_COORDINATES entry '${coord_name}'. Use cartesian_ks or all.")
          endif()
        endforeach()
      endif()
      if(enable_grmhd)
        target_sources(${target_name} PRIVATE
          src/image/grmhd.cpp)
        if(enable_iharm)
          target_sources(${target_name} PRIVATE src/grmhd/iharm_loader.cpp)
        endif()
        if(enable_kharma)
          target_sources(${target_name} PRIVATE src/grmhd/kharma_loader.cpp)
        endif()
        if(enable_athenak)
          target_sources(${target_name} PRIVATE src/grmhd/athenak_loader.cpp)
        endif()
        if(enable_bhac)
          target_sources(${target_name} PRIVATE src/grmhd/bhac_loader.cpp)
        endif()
        if(enable_hamr)
          target_sources(${target_name} PRIVATE
            src/grmhd/bhac_loader.cpp
            src/grmhd/hamr_loader.cpp)
        endif()
        set(requested_grmhd_coordinates ${KPOLARIS_GRMHD_COORDINATES})
        if(requested_grmhd_coordinates STREQUAL "all")
          set(requested_grmhd_coordinates cartesian_ks spherical_ks boyer_lindquist fmks mks)
        endif()
        foreach(coord_name IN LISTS requested_grmhd_coordinates)
          string(TOLOWER "${coord_name}" coord_name_lower)
          if(coord_name_lower STREQUAL "cartesian_ks" OR coord_name_lower STREQUAL "cartesian" OR coord_name_lower STREQUAL "ks_cartesian")
            set(enable_coord_cartesian_ks 1)
            target_sources(${target_name} PRIVATE src/image/backends/grmhd_cartesian_ks.cpp)
          elseif(coord_name_lower STREQUAL "spherical_ks" OR coord_name_lower STREQUAL "ks" OR coord_name_lower STREQUAL "ks_spherical" OR coord_name_lower STREQUAL "sks")
            set(enable_coord_spherical_ks 1)
            target_sources(${target_name} PRIVATE src/image/backends/grmhd_spherical_ks.cpp)
          elseif(coord_name_lower STREQUAL "boyer_lindquist" OR coord_name_lower STREQUAL "bl")
            set(enable_coord_boyer_lindquist 1)
            target_sources(${target_name} PRIVATE src/image/backends/grmhd_bl.cpp)
          elseif(coord_name_lower STREQUAL "fmks")
            set(enable_coord_fmks 1)
            target_sources(${target_name} PRIVATE src/image/backends/grmhd_fmks.cpp)
          elseif(coord_name_lower STREQUAL "mks" OR coord_name_lower STREQUAL "modified_ks" OR coord_name_lower STREQUAL "modified-kerr-schild" OR coord_name_lower STREQUAL "native_mks")
            set(enable_coord_mks 1)
            target_sources(${target_name} PRIVATE src/image/backends/grmhd_mks.cpp)
          else()
            message(FATAL_ERROR "Unknown KPOLARIS_GRMHD_COORDINATES entry '${coord_name}'. Use cartesian_ks, spherical_ks, boyer_lindquist, fmks, mks, or all.")
          endif()
        endforeach()
      endif()
      target_include_directories(${target_name} PRIVATE ${HDF5_INCLUDE_DIRS})
      target_link_libraries(${target_name} PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
      set(require_grmhd_derived_scalars 0)
      if(enable_grmhd AND KPOLARIS_IMAGE_REQUIRE_GRMHD_DERIVED_SCALARS)
        set(require_grmhd_derived_scalars 1)
      endif()
      target_compile_definitions(${target_name} PRIVATE
        KPOLARIS_DEFAULT_MODEL="${model_name_lower}"
        KPOLARIS_DEFAULT_BINARY_RIAF=${enable_binary_riaf}
        KPOLARIS_ENABLE_MODEL_RIAF=${enable_riaf}
        KPOLARIS_ENABLE_MODEL_BINARY_RIAF=${enable_binary_riaf}
        KPOLARIS_ENABLE_MODEL_TORUS=${enable_torus}
        KPOLARIS_ENABLE_MODEL_IHARM=${enable_iharm}
        KPOLARIS_ENABLE_MODEL_KHARMA=${enable_kharma}
        KPOLARIS_ENABLE_MODEL_ATHENAK=${enable_athenak}
        KPOLARIS_ENABLE_MODEL_BHAC=${enable_bhac}
        KPOLARIS_ENABLE_MODEL_HAMR=${enable_hamr}
        KPOLARIS_ENABLE_COORDINATE_CARTESIAN_KS=${enable_coord_cartesian_ks}
        KPOLARIS_ENABLE_COORDINATE_SPHERICAL_KS=${enable_coord_spherical_ks}
        KPOLARIS_ENABLE_COORDINATE_BOYER_LINDQUIST=${enable_coord_boyer_lindquist}
        KPOLARIS_ENABLE_COORDINATE_FMKS=${enable_coord_fmks}
        KPOLARIS_ENABLE_COORDINATE_MKS=${enable_coord_mks}
        KPOLARIS_ENABLE_RIAF_COORDINATE_BOYER_LINDQUIST=${enable_riaf_coord_boyer_lindquist}
        KPOLARIS_ENABLE_TORUS_COORDINATE_CARTESIAN_KS=${enable_torus_coord_cartesian_ks}
        KPOLARIS_RIAF_COMPILED_EMISSION_TYPE=${KPOLARIS_RIAF_EMISSION_TYPE}
        KPOLARIS_IHARM_COMPILED_EMISSION_TYPE=${KPOLARIS_GRMHD_EMISSION_TYPE}
        KPOLARIS_GRMHD_REQUIRE_DERIVED_SCALARS=${require_grmhd_derived_scalars}
        KPOLARIS_ENABLE_CSV_OUTPUT=$<BOOL:${KPOLARIS_ENABLE_CSV_OUTPUT}>
        KPOLARIS_ENABLE_ANALYSIS_MODE=$<BOOL:${KPOLARIS_ENABLE_ANALYSIS_MODE}>
        KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
      list(APPEND KPOLARIS_INSTALL_RUNTIME_TARGETS ${target_name})
    endforeach()
endif()

if(KPOLARIS_BUILD_TRACE_TOOL)
    add_executable(kpolaris_model_trace
      tools/kpolaris_model_trace.cpp
      src/grmhd/kharma_loader.cpp
      src/grmhd/athenak_loader.cpp
      src/grmhd/bhac_loader.cpp
      src/grmhd/hamr_loader.cpp)
    target_compile_definitions(kpolaris_model_trace PRIVATE KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_model_trace_riaf tools/kpolaris_model_trace.cpp)
    target_compile_definitions(kpolaris_model_trace_riaf PRIVATE KPOLARIS_TRACE_ONLY_RIAF=1 KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace_riaf PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace_riaf PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_model_trace_torus tools/kpolaris_model_trace.cpp)
    target_compile_definitions(kpolaris_model_trace_torus PRIVATE KPOLARIS_TRACE_ONLY_TORUS=1 KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace_torus PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace_torus PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_model_trace_athenak
      tools/kpolaris_model_trace.cpp
      src/grmhd/athenak_loader.cpp)
    target_compile_definitions(kpolaris_model_trace_athenak PRIVATE KPOLARIS_TRACE_ONLY_ATHENAK=1 KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace_athenak PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace_athenak PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_model_trace_iharm tools/kpolaris_model_trace.cpp)
    target_compile_definitions(kpolaris_model_trace_iharm PRIVATE KPOLARIS_TRACE_ONLY_IHARM=1 KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace_iharm PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace_iharm PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_model_trace_kharma
      tools/kpolaris_model_trace.cpp
      src/grmhd/kharma_loader.cpp)
    target_compile_definitions(kpolaris_model_trace_kharma PRIVATE KPOLARIS_TRACE_ONLY_KHARMA=1 KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace_kharma PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace_kharma PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_model_trace_bhac
      tools/kpolaris_model_trace.cpp
      src/grmhd/bhac_loader.cpp)
    target_compile_definitions(kpolaris_model_trace_bhac PRIVATE KPOLARIS_TRACE_ONLY_BHAC=1 KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace_bhac PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace_bhac PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    add_executable(kpolaris_model_trace_hamr
      tools/kpolaris_model_trace.cpp
      src/grmhd/bhac_loader.cpp
      src/grmhd/hamr_loader.cpp)
    target_compile_definitions(kpolaris_model_trace_hamr PRIVATE KPOLARIS_TRACE_ONLY_HAMR=1 KPOLARIS_ENABLE_SLOW_LIGHT=$<BOOL:${KPOLARIS_ENABLE_SLOW_LIGHT}>)
    target_include_directories(kpolaris_model_trace_hamr PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(kpolaris_model_trace_hamr PRIVATE kpolaris ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})

    list(APPEND KPOLARIS_INSTALL_RUNTIME_TARGETS
      kpolaris_model_trace
      kpolaris_model_trace_riaf
      kpolaris_model_trace_torus
      kpolaris_model_trace_athenak
      kpolaris_model_trace_iharm
      kpolaris_model_trace_kharma
      kpolaris_model_trace_bhac
      kpolaris_model_trace_hamr)
endif()

if(KPOLARIS_ENABLE_CUDA AND KPOLARIS_INSTALL_RUNTIME_TARGETS)
  find_package(CUDAToolkit REQUIRED)
  # CMake identifies nvcc_wrapper as an NVIDIA C++ compiler and cannot perform
  # its usual post-build RPATH relink for installed executables in that mode.
  # Build these targets with their install RPATH from the outset so enabling
  # installation never makes an otherwise valid CUDA configuration fail at
  # generate time. Retain the external dependency search paths in that RPATH:
  # otherwise a user-space CUDA/HDF5 installation can compile successfully but
  # the executable cannot find its shared libraries when launched.
  set_target_properties(${KPOLARIS_INSTALL_RUNTIME_TARGETS}
    PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE INSTALL_RPATH_USE_LINK_PATH TRUE)
  # nvcc's implicit link directories are omitted by USE_LINK_PATH even when
  # the dynamic loader does not know them. Add only the actual CUDA runtime
  # directory, never the link-only CUDA driver stub directory.
  set_property(TARGET ${KPOLARIS_INSTALL_RUNTIME_TARGETS} APPEND PROPERTY
    INSTALL_RPATH "$<TARGET_FILE_DIR:CUDA::cudart>")
endif()

install(TARGETS kpolaris EXPORT KPolarisTargets)
if(KPOLARIS_INSTALL_RUNTIME_TARGETS)
  install(TARGETS ${KPOLARIS_INSTALL_RUNTIME_TARGETS}
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" OPTIONAL)
endif()

install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kpolaris"
  FILES_MATCHING PATTERN "*.hpp"
  PATTERN "binary_trajectory_hdf5.hpp" EXCLUDE)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/common/version.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kpolaris/common")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/params/"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/kpolaris/params"
  FILES_MATCHING PATTERN "*.par")

set(KPOLARIS_INSTALL_ANALYSIS_SCRIPTS
  scripts/analyze_closure_stokes.py
  scripts/analyze_physical_responses.py
  scripts/analyze_polarization_budget.py
  scripts/analyze_trace_mechanisms.py
  scripts/benchmark_image_scaling.py
  scripts/bin_trace_physical_maps.py
  scripts/compare_analysis_trace.py
  scripts/compare_slow_light_batches.py
  scripts/compare_stokes_csv.py
  scripts/compare_stokes_hdf5.py
  scripts/export_wiki.py
  scripts/kpolaris.py
  scripts/kpolaris_ddc_parallel.py
  scripts/kpolaris_ddc_reconstruct.py
  scripts/kpolaris_evpa.py
  scripts/kpolaris_image.py
  scripts/plot_analysis_maps.py
  scripts/plot_direct_only_compare.py
  scripts/plot_equatorial_compare.py
  scripts/plot_kpolaris_pol.py
  scripts/plot_physical_response_gallery.py
  scripts/plot_stokes_compare.py
  scripts/plot_trace_diagnostics.py
  scripts/prepare_kharma_cache.py
  scripts/serve_ddc_parallel.py
  scripts/generate_binary_trajectory.py)
install(PROGRAMS ${KPOLARIS_INSTALL_ANALYSIS_SCRIPTS}
  DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/kpolaris")
install(FILES README.md README_zh.md CHANGELOG.md CITATION.cff LICENSE CONTRIBUTING.md
  DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/KPolaris")
install(DIRECTORY docs/ DESTINATION "${CMAKE_INSTALL_DATADIR}/doc/KPolaris/docs")

set(KPOLARIS_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/KPolaris")
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/KPolarisConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/KPolarisConfig.cmake"
  INSTALL_DESTINATION "${KPOLARIS_INSTALL_CMAKEDIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/KPolarisConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMinorVersion)
install(EXPORT KPolarisTargets
  FILE KPolarisTargets.cmake
  NAMESPACE KPolaris::
  DESTINATION "${KPOLARIS_INSTALL_CMAKEDIR}")
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/KPolarisConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/KPolarisConfigVersion.cmake"
  DESTINATION "${KPOLARIS_INSTALL_CMAKEDIR}")

if(KPOLARIS_BUILD_TESTS)
  if(KPOLARIS_ENABLE_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Linux"
      AND NOT CMAKE_SKIP_RPATH AND NOT CMAKE_SKIP_INSTALL_RPATH
      AND TARGET kpolaris_model_image_riaf)
    # --version runs before GPU initialization, but still exercises the ELF
    # loader. Do not let an activated environment hide missing runtime RPATHs.
    add_test(NAME kpolaris_cuda_runtime_loading
      COMMAND "${CMAKE_COMMAND}" -E env --unset=LD_LIBRARY_PATH
        $<TARGET_FILE:kpolaris_model_image_riaf> --version)
    set_tests_properties(kpolaris_cuda_runtime_loading PROPERTIES TIMEOUT 30)
  endif()
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c "import h5py, numpy"
      RESULT_VARIABLE KPOLARIS_TRAJECTORY_PYTHON_DEPS_RESULT
      OUTPUT_QUIET
      ERROR_QUIET)
    if(KPOLARIS_TRAJECTORY_PYTHON_DEPS_RESULT EQUAL 0)
      if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        add_test(NAME kpolaris_image_benchmark_runner
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_image_benchmark_runner.py")
        set_tests_properties(kpolaris_image_benchmark_runner PROPERTIES TIMEOUT 30)
      endif()
      if(TARGET kpolaris_model_image_riaf)
        add_test(NAME kpolaris_repeat_images
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_repeat_images.py"
            --image-exe=$<TARGET_FILE:kpolaris_model_image_riaf>
            --parameters=${CMAKE_CURRENT_SOURCE_DIR}/params/riaf_recommended.par
            --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_repeat_images)
        set_tests_properties(kpolaris_repeat_images PROPERTIES TIMEOUT 120)
      endif()
      add_test(NAME kpolaris_trace_mechanisms
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_trace_mechanisms.py")
      add_test(NAME kpolaris_physical_response_python
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_physical_response.py")
      foreach(response_model riaf iharm athenak)
        if(TARGET kpolaris_model_image_${response_model} AND KPOLARIS_ENABLE_ANALYSIS_MODE AND KPOLARIS_ENABLE_SLOW_LIGHT)
          add_test(NAME kpolaris_equatorial_${response_model}
            COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_equatorial.py"
              --${response_model}=$<TARGET_FILE:kpolaris_model_image_${response_model}>
              --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_equatorial_${response_model})
          add_test(NAME kpolaris_direct_only_${response_model}
            COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_direct_only.py"
              --${response_model}=$<TARGET_FILE:kpolaris_model_image_${response_model}>
              --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_direct_only_${response_model})
          add_test(NAME kpolaris_physical_response_${response_model}
            COMMAND "${Python3_EXECUTABLE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_physical_response.py"
              --${response_model}=$<TARGET_FILE:kpolaris_model_image_${response_model}>
              --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_physical_response_${response_model})
        endif()
      endforeach()
      if(KPOLARIS_ENABLE_SLOW_LIGHT AND NOT KPOLARIS_USE_FLOAT)
        foreach(slow_model iharm kharma athenak bhac hamr)
          if(TARGET kpolaris_model_image_${slow_model})
            add_test(NAME kpolaris_slow_light_unified_${slow_model}
              COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_slow_light_unified.py"
                --image-exe=$<TARGET_FILE:kpolaris_model_image_${slow_model}>
                --model=${slow_model} --analysis-enabled=$<BOOL:${KPOLARIS_ENABLE_ANALYSIS_MODE}>
                --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_slow_light_unified_${slow_model})
            set_tests_properties(kpolaris_slow_light_unified_${slow_model} PROPERTIES TIMEOUT 240)
          endif()
        endforeach()
      endif()
      if(TARGET kpolaris_model_image_kharma AND KPOLARIS_ENABLE_SLOW_LIGHT AND NOT KPOLARIS_USE_FLOAT)
        add_test(NAME kpolaris_slow_light_decoupled_cli
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_slow_light_decoupled_cli.py"
            --image-exe=$<TARGET_FILE:kpolaris_model_image_kharma>
            --analysis-enabled=$<BOOL:${KPOLARIS_ENABLE_ANALYSIS_MODE}>
            --max-frequencies=${KPOLARIS_MAX_FREQUENCIES}
            --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_slow_light_decoupled_cli)
        set_tests_properties(kpolaris_slow_light_decoupled_cli PROPERTIES TIMEOUT 180)
      endif()
      if(TARGET kpolaris_model_image_kharma AND KPOLARIS_ENABLE_ANALYSIS_MODE AND KPOLARIS_ENABLE_SLOW_LIGHT AND NOT KPOLARIS_USE_FLOAT)
        add_test(NAME kpolaris_ddc_parallel_scheduler
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ddc_parallel.py")
        set_tests_properties(kpolaris_ddc_parallel_scheduler PROPERTIES TIMEOUT 30)
        add_test(NAME kpolaris_ddc_reconstruction
          COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ddc_reconstruct.py")
        set_tests_properties(kpolaris_ddc_reconstruction PROPERTIES TIMEOUT 30)
        add_test(NAME kpolaris_slow_light_batch_cli
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_slow_light_batch_cli.py"
            --image-exe=$<TARGET_FILE:kpolaris_model_image_kharma>
            --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_slow_light_batch_cli)
        set_tests_properties(kpolaris_slow_light_batch_cli PROPERTIES TIMEOUT 180)
        add_test(NAME kpolaris_fluid_time_interpolation_cli
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_slow_light_batch_cli.py"
            --image-exe=$<TARGET_FILE:kpolaris_model_image_kharma>
            --interpolation=fluid
            --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_fluid_time_interpolation_cli)
        set_tests_properties(kpolaris_fluid_time_interpolation_cli PROPERTIES TIMEOUT 180)
        add_test(NAME kpolaris_coefficient_time_interpolation_cli
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_slow_light_batch_cli.py"
            --image-exe=$<TARGET_FILE:kpolaris_model_image_kharma>
            --interpolation=coefficients
            --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_coefficient_time_interpolation_cli)
        set_tests_properties(kpolaris_coefficient_time_interpolation_cli PROPERTIES TIMEOUT 180)
        if(KPOLARIS_MAX_FREQUENCIES GREATER_EQUAL 2)
          set(kpolaris_interpolation_path_args
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_slow_light_interpolation_paths.py"
            --image-exe=$<TARGET_FILE:kpolaris_model_image_kharma>
            --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_slow_light_interpolation_paths)
          if(TARGET kpolaris_model_trace_kharma)
            list(APPEND kpolaris_interpolation_path_args --trace-exe=$<TARGET_FILE:kpolaris_model_trace_kharma>)
          endif()
          add_test(NAME kpolaris_slow_light_interpolation_paths
            COMMAND "${Python3_EXECUTABLE}" ${kpolaris_interpolation_path_args})
          set_tests_properties(kpolaris_slow_light_interpolation_paths PROPERTIES TIMEOUT 120)
        endif()
      endif()
      if(UNIX AND TARGET kpolaris_model_image_kharma AND KPOLARIS_ENABLE_ANALYSIS_MODE AND KPOLARIS_ENABLE_SLOW_LIGHT AND KPOLARIS_MAX_FREQUENCIES GREATER_EQUAL 2)
        set(kpolaris_ddc_test_args
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ddc.py"
          --image=$<TARGET_FILE:kpolaris_model_image_kharma>
          --workdir=${CMAKE_CURRENT_BINARY_DIR}/test_ddc)
        if(TARGET kpolaris_model_trace_kharma)
          list(APPEND kpolaris_ddc_test_args --trace=$<TARGET_FILE:kpolaris_model_trace_kharma>)
        endif()
        add_test(NAME kpolaris_ddc_input COMMAND "${Python3_EXECUTABLE}" ${kpolaris_ddc_test_args})
        set_tests_properties(kpolaris_ddc_input PROPERTIES TIMEOUT 600)
      endif()
      add_test(NAME kpolaris_polarization_budget
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_polarization_budget.py")
      add_test(NAME kpolaris_binary_trajectory_generator
        COMMAND "${Python3_EXECUTABLE}"
          "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_generate_binary_trajectory.py")
    else()
      message(WARNING
        "Skipping binary trajectory generator CTest because Python "
        "dependencies numpy and h5py are unavailable to "
        "${Python3_EXECUTABLE}")
    endif()

    if(KPOLARIS_FETCH_KOKKOS)
      set(kpolaris_test_kokkos_dir "${kokkos_BINARY_DIR}")
    else()
      set(kpolaris_test_kokkos_dir "${Kokkos_DIR}")
    endif()
    if(EXISTS "${kpolaris_test_kokkos_dir}/KokkosConfig.cmake")
      set(kpolaris_install_test_args
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_installation.py"
        "--cmake" "${CMAKE_COMMAND}"
        "--build-dir" "${CMAKE_CURRENT_BINARY_DIR}"
        "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/test_installation"
        "--expected-version" "${PROJECT_VERSION}"
        "--config" "${CMAKE_BUILD_TYPE}"
        "--generator" "${CMAKE_GENERATOR}"
        "--make-program" "${CMAKE_MAKE_PROGRAM}"
        "--cxx-compiler" "${CMAKE_CXX_COMPILER}"
        "--kokkos-dir" "${kpolaris_test_kokkos_dir}"
        "--bindir" "${CMAKE_INSTALL_BINDIR}"
        "--includedir" "${CMAKE_INSTALL_INCLUDEDIR}"
        "--libdir" "${CMAKE_INSTALL_LIBDIR}"
        "--libexecdir" "${CMAKE_INSTALL_LIBEXECDIR}"
        "--datadir" "${CMAKE_INSTALL_DATADIR}")
      if(KPOLARIS_ENABLE_CUDA AND KPOLARIS_FETCH_KOKKOS)
        set(kpolaris_test_kokkos_compile_launcher
          "${kokkos_SOURCE_DIR}/bin/kokkos_launch_compiler")
        set(kpolaris_test_kokkos_nvcc_wrapper
          "${kokkos_SOURCE_DIR}/bin/nvcc_wrapper")
        if(EXISTS "${kpolaris_test_kokkos_compile_launcher}")
          list(APPEND kpolaris_install_test_args
            "--kokkos-compile-launcher"
            "${kpolaris_test_kokkos_compile_launcher}")
        endif()
        if(EXISTS "${kpolaris_test_kokkos_nvcc_wrapper}")
          list(APPEND kpolaris_install_test_args
            "--kokkos-nvcc-wrapper"
            "${kpolaris_test_kokkos_nvcc_wrapper}")
        endif()
      endif()
      set(kpolaris_install_runtime_library_dirs)
      foreach(dependency_library IN LISTS HDF5_CXX_LIBRARIES HDF5_LIBRARIES)
        if(IS_ABSOLUTE "${dependency_library}")
          get_filename_component(dependency_library_dir
            "${dependency_library}" DIRECTORY)
          list(APPEND kpolaris_install_runtime_library_dirs
            "${dependency_library_dir}")
        elseif(TARGET "${dependency_library}")
          get_target_property(dependency_imported_configurations
            "${dependency_library}" IMPORTED_CONFIGURATIONS)
          foreach(dependency_configuration IN LISTS
              dependency_imported_configurations)
            string(TOUPPER "${dependency_configuration}"
              dependency_configuration_upper)
            get_target_property(dependency_imported_location
              "${dependency_library}"
              "IMPORTED_LOCATION_${dependency_configuration_upper}")
            if(IS_ABSOLUTE "${dependency_imported_location}")
              get_filename_component(dependency_library_dir
                "${dependency_imported_location}" DIRECTORY)
              list(APPEND kpolaris_install_runtime_library_dirs
                "${dependency_library_dir}")
            endif()
          endforeach()
        endif()
      endforeach()
      list(APPEND kpolaris_install_runtime_library_dirs
        ${HDF5_LIBRARY_DIRS}
        ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES})
      list(REMOVE_DUPLICATES kpolaris_install_runtime_library_dirs)
      foreach(runtime_library_dir IN LISTS kpolaris_install_runtime_library_dirs)
        list(APPEND kpolaris_install_test_args
          "--runtime-library-dir" "${runtime_library_dir}")
      endforeach()
      foreach(runtime_target IN LISTS KPOLARIS_INSTALL_RUNTIME_TARGETS)
        list(APPEND kpolaris_install_test_args
          "--runtime-name" "$<TARGET_FILE_NAME:${runtime_target}>")
      endforeach()
      add_test(NAME kpolaris_installation
        COMMAND "${Python3_EXECUTABLE}" ${kpolaris_install_test_args})
      set_tests_properties(kpolaris_installation PROPERTIES RUN_SERIAL TRUE)

      if(NOT KPOLARIS_ENABLE_SANITIZERS AND NOT KPOLARIS_ENABLE_CUDA)
        add_test(NAME kpolaris_source_archive_build
          COMMAND "${Python3_EXECUTABLE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_source_archive_build.py"
            "--source-dir" "${CMAKE_CURRENT_SOURCE_DIR}"
            "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/test_source_archive_build"
            "--cmake" "${CMAKE_COMMAND}"
            "--generator" "${CMAKE_GENERATOR}"
            "--make-program" "${CMAKE_MAKE_PROGRAM}"
            "--cxx-compiler" "${CMAKE_CXX_COMPILER}"
            "--kokkos-dir" "${kpolaris_test_kokkos_dir}"
            "--expected-version" "${PROJECT_VERSION}")
        set_tests_properties(kpolaris_source_archive_build PROPERTIES RUN_SERIAL TRUE)
      endif()
    else()
      message(WARNING
        "Skipping install/source-archive CTest registration because a reusable "
        "KokkosConfig.cmake was not found at '${kpolaris_test_kokkos_dir}'")
    endif()

    add_test(NAME kpolaris_user_workflows
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_user_workflows.py")

    add_test(NAME kpolaris_release_metadata_contract
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_release_metadata.py")
    add_test(NAME kpolaris_wiki_export
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_wiki_export.py")
    add_test(NAME kpolaris_release_metadata
      COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/scripts/check_release_metadata.py"
        "--source-dir" "${CMAKE_CURRENT_SOURCE_DIR}"
        "--expected-version" "${PROJECT_VERSION}")
  endif()
  if(Python3_Interpreter_FOUND AND TARGET kpolaris_model_image_riaf)
    string(REPLACE ";" "," kpolaris_hdf5_grmhd_coordinates
      "${KPOLARIS_GRMHD_COORDINATES}")
    set(kpolaris_hdf5_contract_args
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_hdf5_contract.py"
      "--image-exe" "$<TARGET_FILE:kpolaris_model_image_riaf>"
      "--plot-script" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_kpolaris_pol.py"
      "--analysis-script" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_analysis_maps.py"
      "--analysis-compare-script" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_analysis_trace.py"
      "--compare-script" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/compare_stokes_hdf5.py"
      "--trace-diagnostics-script" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_trace_diagnostics.py"
      "--trace-bins-script" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/bin_trace_physical_maps.py"
      "--compiled-riaf-emission-type" "${KPOLARIS_RIAF_EMISSION_TYPE}"
      "--compiled-grmhd-coordinates" "${kpolaris_hdf5_grmhd_coordinates}"
      "--expected-version" "${PROJECT_VERSION}"
      "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/test_hdf5_contract")
    if(TARGET kpolaris_model_trace)
      list(APPEND kpolaris_hdf5_contract_args
        "--trace-exe" "$<TARGET_FILE:kpolaris_model_trace>")
    endif()
    if(KPOLARIS_ENABLE_CSV_OUTPUT)
      list(APPEND kpolaris_hdf5_contract_args "--csv-output-enabled")
    endif()
    if(TARGET kpolaris_model_image_torus)
      list(APPEND kpolaris_hdf5_contract_args
        "--torus-image-exe" "$<TARGET_FILE:kpolaris_model_image_torus>"
        "--torus-parameter-file" "${CMAKE_CURRENT_SOURCE_DIR}/params/torus_recommended.par")
    endif()
    if(TARGET kpolaris_model_image_iharm)
      list(APPEND kpolaris_hdf5_contract_args
        "--iharm-image-exe" "$<TARGET_FILE:kpolaris_model_image_iharm>")
    endif()
    if(TARGET kpolaris_model_image_kharma)
      list(APPEND kpolaris_hdf5_contract_args
        "--kharma-image-exe" "$<TARGET_FILE:kpolaris_model_image_kharma>")
    endif()
    if(TARGET kpolaris_model_image_athenak)
      list(APPEND kpolaris_hdf5_contract_args
        "--athenak-image-exe" "$<TARGET_FILE:kpolaris_model_image_athenak>")
    endif()
    if(TARGET kpolaris_model_image_bhac)
      list(APPEND kpolaris_hdf5_contract_args
        "--bhac-image-exe" "$<TARGET_FILE:kpolaris_model_image_bhac>")
    endif()
    if(TARGET kpolaris_model_image_hamr)
      list(APPEND kpolaris_hdf5_contract_args
        "--hamr-image-exe" "$<TARGET_FILE:kpolaris_model_image_hamr>")
    endif()
    set(kpolaris_evpa_test_args "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_evpa_output.py"
      "--image-exe" "$<TARGET_FILE:kpolaris_model_image_riaf>"
      "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/test_evpa_output")
    if(TARGET kpolaris_model_trace)
      list(APPEND kpolaris_evpa_test_args "--trace-exe" "$<TARGET_FILE:kpolaris_model_trace>")
    endif()
    if(KPOLARIS_ENABLE_ANALYSIS_MODE)
      list(APPEND kpolaris_evpa_test_args "--analysis-enabled")
    endif()
    if(KPOLARIS_ENABLE_CSV_OUTPUT)
      list(APPEND kpolaris_evpa_test_args "--csv-enabled")
    endif()
    if(KPOLARIS_TRAJECTORY_PYTHON_DEPS_RESULT EQUAL 0)
      add_test(NAME kpolaris_evpa_output
        COMMAND "${Python3_EXECUTABLE}" ${kpolaris_evpa_test_args})
    endif()
    add_test(NAME kpolaris_hdf5_contract
      COMMAND "${Python3_EXECUTABLE}" ${kpolaris_hdf5_contract_args})
    set_tests_properties(kpolaris_hdf5_contract PROPERTIES SKIP_RETURN_CODE 77)

    set(kpolaris_loader_fuzz_args
      "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_loader_fuzz.py"
      "--workdir" "${CMAKE_CURRENT_BINARY_DIR}/test_loader_fuzz"
      "--cases-per-loader" "12"
      "--seed" "20260716"
      "--timeout" "30")
    set(kpolaris_have_loader_fuzz_target FALSE)
    foreach(kpolaris_loader_name iharm kharma athenak bhac hamr)
      if(TARGET kpolaris_model_image_${kpolaris_loader_name})
        list(APPEND kpolaris_loader_fuzz_args
          "--${kpolaris_loader_name}-image-exe"
          "$<TARGET_FILE:kpolaris_model_image_${kpolaris_loader_name}>")
        set(kpolaris_have_loader_fuzz_target TRUE)
      endif()
    endforeach()
    if(kpolaris_have_loader_fuzz_target)
      add_test(NAME kpolaris_loader_fuzz
        COMMAND "${Python3_EXECUTABLE}" ${kpolaris_loader_fuzz_args})
      set_tests_properties(kpolaris_loader_fuzz PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 300)
    endif()
  endif()
endif()
