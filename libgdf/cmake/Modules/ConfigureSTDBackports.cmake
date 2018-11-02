#=============================================================================
# Copyright 2018 BlazingDB, Inc.
# Contributors:
#     Eyal Rozenberg <eyalroz@blazingdb.com>
#
# This file may only be used under the terms of the Apache License, version
# 2.0. A copy of the license may be obtained at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#=============================================================================

set(BASE_DIR_FOR_THIRD_PARTY_CODE "${CMAKE_SOURCE_DIR}/thirdparty")

ExternalProject_Add(mpark_variant_project                            # Name for custom target
	PREFIX          CMakeFiles/mpark_variant_project                 # Root dir for entire project
	TMP_DIR         CMakeFiles/mpark_variant_project/tmp             # Directory to store temporary files
	STAMP_DIR       CMakeFiles/mpark_variant_project/stamp           # Directory to store step timestamps
	GIT_REPOSITORY  git@github.com:mpark/variant.git                 # URL of git repo
	GIT_TAG         ab86b12bee3cd8895bf7b1f1cea1c1ccca1cfcce         # Git branch name, commit id or tag
	UPDATE_COMMAND  ""                                               # Source work-tree update command
	SOURCE_DIR      "${BASE_DIR_FOR_THIRD_PARTY_CODE}/variant"       # Source dir to be used for build
	BUILD_IN_SOURCE 1                                                # Use source dir (of external project) as its build dir
	BUILD_COMMAND   ":"                                              # <- This will prevent building.
	INSTALL_COMMAND ""                                               # Command to drive install after build
)
ExternalProject_Get_Property(mpark_variant_project SOURCE_DIR)
include_directories(${SOURCE_DIR}/include)

ExternalProject_Add(tl_optional_project                              # Name for custom target
	PREFIX          CMakeFiles/tl_optional_project                   # Root dir for entire project
	TMP_DIR         CMakeFiles/tl_optional_project/tmp               # Directory to store temporary files
	STAMP_DIR       CMakeFiles/tl_optional_project/stamp             # Directory to store step timestamps
	GIT_REPOSITORY  git@github.com:TartanLlama/optional.git          # URL of git repo
	GIT_TAG         3449fbc904dcfa5738905befd8114bdfda82f1ec         # Git branch name, commit id or tag
	UPDATE_COMMAND  ""                                               # Source work-tree update command
	SOURCE_DIR      "${BASE_DIR_FOR_THIRD_PARTY_CODE}/optional"      # Source dir to be used for build
	BUILD_IN_SOURCE 1                                                # Use source dir (of external project) as its build dir
	BUILD_COMMAND   ":"                                              # <- This will prevent building.
	INSTALL_COMMAND ""                                               # Command to drive install after build
)
ExternalProject_Get_Property(tl_optional_project SOURCE_DIR)
include_directories(${SOURCE_DIR})

# TODO:
#
# Instead of just adding include directories, we could use these projects' `CMakeLists.txt`s - which define targets to depend on

