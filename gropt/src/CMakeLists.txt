cmake_minimum_required(VERSION 3.10.0)
project(gropt VERSION 0.0.0 LANGUAGES CXX)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/core)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/operators)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/solvers)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/external)

add_definitions(-DFMT_UNICODE=0)
add_definitions(-DH5_USE_EIGEN)

add_subdirectory(external/matplotplusplus)
add_subdirectory(external/highfive)

set(SRC_FILES
    core/fft_helper.cpp
    core/gropt_params.cpp
    core/gropt_utils.cpp
    solvers/equilibrate.cpp
    solvers/ils.cpp
    solvers/ils_cg.cpp
    solvers/ils_nlcg.cpp
    solvers/ils_bicgstabl.cpp
    solvers/solver.cpp
    solvers/solver_groptsdmm.cpp
    solvers/solver_osqp.cpp
    solvers/workspace_osqp.cpp
    solvers/workspace_sdmm.cpp
    solvers/workspace_solver.cpp
    operators/op_bvalue.cpp
    operators/op_gradient.cpp
    operators/op_identity.cpp
    operators/op_main.cpp
    operators/op_moment.cpp
    operators/op_acoustic.cpp
    operators/op_slew.cpp
    operators/op_safe.cpp
    operators/op_eddy.cpp
    operators/op_tv.cpp
)

add_executable(gropt main.cpp ${SRC_FILES})
