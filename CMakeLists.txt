# CMakeList.txt : projet CMake pour VolumeComputation, incluez la source et définissez
# la logique spécifique au projet ici.
#
project(PDMP)

cmake_minimum_required (VERSION 3.8)

cmake_minimum_required (VERSION 3.8)

list(APPEND CMAKE_MODULE_PATH "C:/Users/gus_5/Documents/PostDoc/Eigen3/install")

set(CMAKE_CXX_STANDARD 20)

#project ("VolumeComputation" LANGUAGES CXX CUDA )
project ("VolumeComputation" LANGUAGES CXX)


# add_compile_options("-W" "-Wall" "-O0" "-g" "-pedantic")
#add_compile_options("-W" "-Wall" "-O3" "-pedantic" "-march=native")
add_compile_options("-W" "-O3" "-march=native" "-g")


# I stole FindMPFR.cmake from the SBL: we need to find MPFR below

#set(CMAKE_MODULE_PATH /home/fcazals/projects/proj-research/git/augustin-papers/PDMP-convex/code)

#WINDOWS BOOST
#set(Boost_INCLUDE_DIR "C:\\boost_1_76_0")


set(Eigen3_DIR "C:\\Users\\gus_5\\Documents\\PostDoc\\Eigen3\\install")
#set(EIGEN_USE_BLAS 1)
#INCLUDE_DIRECTORIES ( "C:\\Users\\gus_5\\Documents\\PostDoc\\Eigen3" )

#find_package(CUDA REQUIRED cublas)


#find_package(Thrust REQUIRED)
#thrust_create_target(Thrust)


#to activate BLAS, uncomment this
#set(BLAS_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3")
#set(BLAS_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3")
#set(BLAS_INCDIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.3/include")
#find_package(BLAS REQUIRED)


#find_package(BLAS REQUIRED)
# xfc: variables for Eigen3
find_package (Eigen3 REQUIRED NO_MODULE) 
#if(Eigen3_FOUND OR Eigen_FOUND)
#  message(STATUS "Eigen library found : ${EIGEN3_INCLUDE_DIRS}")
#  include_directories(${EIGEN3_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS}/..)
#endif()

# xfc boost static
# set(CGAL_Boost_USE_STATIC_LIBS ON)
find_package(CGAL)


#FIND_PACKAGE(Boost 1.6 REQUIRED program_options)
FIND_PACKAGE(Boost 1.6 REQUIRED program_options)

#find_package(Boost COMPONENTS regex program_options)

#LINK_DIRECTORIES(${Boost_LIBRARY_DIRS} ${CUDA_LIBRARY_DIR})
#message("${BLA_VENDOR}")
set(Boost_USE_STATIC_LIBS ON)
find_package(OpenMP)

find_package(GMP REQUIRED)
include_directories(${GOBJECT_INCLUDE_DIR})
find_package(MPFI REQUIRED)

#  xfc: variables for MPFR
# xfc2ac: attention, I need the find_package(MPFR) on mouflage
find_package(MPFR)
if(MPFR_FOUND)
   message(STATUS "MPFR library found: ${MPFR_LIBRARIES_DIR}")
   include_directories("${MPFR_INCLUDE_DIR}")
   message("${MPFR_INCLUDE_DIR}")
   link_directories("${MPFR_LIBRARIES_DIR}")
   #link_directories("${GMP_LIBRARIES_DIR}")
   message("${MPFR_LIBRARIES_DIR}")
 endif()

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif() 

# Incluez les sous-projets.
#add_subdirectory ("VolumeComputation")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")


#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
#SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
#SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
# Ajoutez une source à l'exécutable de ce projet.

#for CUDA, enable this
#set(THRUST_DEVICE_SYSTEM THRUST_DEVICE_SYSTEM_CUDA)
#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${CUDA_ARCH} -O3 -g --std=c++17 -arch=compute_75)


#add_library(cudaImpl STATIC cudaBPS.cu )
#set_target_properties( cudaImpl
#                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)è
#add_executable (VolumeComputation.exe VolumeComputation.cpp cudaHelper.cu)
add_executable (VolumeComputation.exe VolumeComputation.cpp)
add_executable (FigureESS.exe FigureESS.cpp)
#set_property(TARGET VolumeComputation.exe PROPERTY CUDA_ARCHITECTURES 75)#add_executable (FigureESS "FigureESS.cpp")
#target_link_libraries( FigureESS ${Eigen_LIBS} )

target_link_libraries( FigureESS.exe Eigen3::Eigen Boost::boost mpfr)

# xfc
target_link_libraries (VolumeComputation.exe ${GOBJECT_LIBRARIES} Eigen3::Eigen mpfr Boost::boost ${Boost_LIBRARIES} gmp mpfi)#gmp libgmp)
#target_link_libraries (VolumeComputation.exe Eigen3::Eigen mpfr blas Boost::boost)  #${Boost_LIBRARIES}

#GPU
#target_link_libraries (VolumeComputation.exe Eigen3::Eigen Boost::boost ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
target_link_libraries (FigureESS.exe Eigen3::Eigen  Boost::boost)
#BLAS
#target_link_libraries (FigureESS Eigen3::Eigen ${Boost_LIBRARIES} Boost::boost mpfr blas)

#target_link_libraries(Volum PRIVATE Boost::boost)
#target_link_libraries(VolumeComputation LINK_PUBLIC ${Boost_LIBRARIES} )
# TODO: Ajoutez des tests et installez des cibles si nécessaire.
