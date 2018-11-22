cmake_minimum_required(VERSION 2.8.3)

project(svo)

SET(EIGEN_MALLOC_ALREADY_ALIGNED 0)
set( CMAKE_CXX_COMPILER "g++" )
#set( CMAKE_BUILD_TYPE "RelWithDebInfo" )
set( CMAKE_BUILD_TYPE "Debug" )
#set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -Wall -o2 -fopenmp" )
#set( CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3 -fopenmp -Wall" )


list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# Eigen
include_directories( "/usr/include/eigen3" )


# OpenCV
#SET(OpenCV_DIR /usr/share/OpenCV)
SET(OpenCV_DIR ${PROJECT_SOURCE_DIR}/../../opencv/build)
find_package(OpenCV 2 REQUIRED)

# Pangolin
#SET(Pangolin_DIR ${PROJECT_SOURCE_DIR}/../Pangolin/build)
find_package(Pangolin REQUIRED)

# Sophus
#SET(Sophus_DIR ${PROJECT_SOURCE_DIR}/../Sophus/build)
find_package( Sophus REQUIRED )
include_directories( ${Sophus_INCLUDE_DIRS} )
# G2O
#SET(G2O_DIR ${PROJECT_SOURCE_DIR}/../g2o/build)
find_package( G2O REQUIRED )
include_directories( ${G2O_INCLUDE_DIRS} )


include_directories( ${OpenCV_INCLUDE_DIRS} ${Pangolin_INCLUDE_DIRS})


include_directories(include five-point-nister ${EIGEN3_INCLUDE_DIR})

#file(GLOB_RECURSE FIVE_POINT_SOURCE five-point-nister/*.cpp)
#add_library(five-point-nister SHARED ${FIVE_POINT_SOURCE})

#build the tracking node
file(GLOB_RECURSE VO_SOURCE 
          src/backend/*.cpp src/frontend/*.cpp src/logic/*.cpp)

#add_library(feature_tracking_hidden SHARED
#        src/logic/track.cpp
#)


set( THIRD_PARTY_LIBS
        ${Sophus_LIBRARIES}
        g2o_core g2o_stuff g2o_types_sba
        )



#add_executable(vo_online  ${VO_SOURCE} src/online/main.cpp)
add_executable(vo_offline  ${VO_SOURCE} src/offline/main.cpp)
add_executable(five_points_test test/testFivePoints.cpp)
add_executable(test_Essential test/testEssential.cpp)
add_executable(pnp_test test/testPnP.cpp)
add_executable(featuredetector test/featuredetector.cpp  )
add_executable(klt test/KLT.cpp  )



#message(STATUS ${OpenCV_INCLUDE_DIRS} ${OpenCV_LIBS})
#link libriaries
#target_link_libraries(five-point-nister ${OpenCV_LIBS})
#target_link_libraries(vo_online ${OpenCV_LIBS}  five-point-nister
#      ${Pangolin_LIBRARIES}  ueye_api  ${THIRD_PARTY_LIBS}
#        ${PROJECT_SOURCE_DIR}/src/logic/libfeature_tracking_hidden.so
#        )
target_link_libraries(vo_offline ${OpenCV_LIBS}
      ${Pangolin_LIBRARIES}   ${THIRD_PARTY_LIBS}
        ${PROJECT_SOURCE_DIR}/src/logic/libfeature_tracking_hidden.so
	${PROJECT_SOURCE_DIR}/src/logic/libfive-point-nister.so
        )
target_link_libraries(five_points_test ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/src/logic/libfive-point-nister.so)
target_link_libraries(test_Essential ${OpenCV_LIBS} ${PROJECT_SOURCE_DIR}/src/logic/libfive-point-nister.so )
target_link_libraries(pnp_test ${PROJECT_SOURCE_DIR}/src/logic/libfive-point-nister.so ${OpenCV_LIBS} )
target_link_libraries(featuredetector ${OpenCV_LIBS} )
target_link_libraries(klt ${OpenCV_LIBS} )
