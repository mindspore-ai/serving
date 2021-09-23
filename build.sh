#!/bin/bash
set -e
PROJECTPATH=$(cd "$(dirname $0)"; pwd)
export BUILD_PATH="${PROJECTPATH}/build/"

# print usage message
usage()
{
  echo "Usage:"
  echo "    bash build.sh -p {mindspore_shared_lib}] [-j[n]] [-d] [-S on|off] "
  echo "    bash build.sh -e gpu|ascend|cpu|npu [-V 9.2|10.1|310|910] [-j[n]] [-d] [-S on|off] "
  echo "    bash build.sh -t on [-j[n]] [-d] [-S on|off] "
  echo ""
  echo "Options:"
  echo "    -p MindSpore lib {mindspore_shared_lib} path."
  echo "    -e Use cpu, gpu, npu or ascend"
  echo "    -V Specify the device version, if -e gpu, default CUDA 10.1, if -e ascend, default Ascend 910"
  echo "    -j[n] Set the threads when building (Default: -j8)"
  echo "    -d Debug model"
  echo "    -t Build testcases."
  echo "    -S Enable enable download cmake compile dependency from gitee , default off"
}

# check value of input is 'on' or 'off'
# usage: check_on_off arg_value arg_name
check_on_off()
{
  if [[ "X$1" != "Xon" && "X$1" != "Xoff" ]]; then
    echo "Invalid value $1 for option -$2"
    usage
    exit 1
  fi
}

# check and set options
checkopts()
{
  # Init default values of build options
  THREAD_NUM=8
  VERBOSE=""
  DEBUG_MODE="off"
  ENABLE_COVERAGE="off"
  ENABLE_ASAN="off"
  ENABLE_PYTHON="on"
  MS_WHL_LIB_PATH=""
  MS_BACKEND=""
  MS_VERSION=""
  RUN_TESTCASES="off"
  ENABLE_GITEE="off"

  # Process the options
  while getopts 'dvc:j:a:p:e:V:t:S:' opt
  do
    LOW_OPTARG=$(echo ${OPTARG} | tr '[A-Z]' '[a-z]')

    case "${opt}" in
      e)
        echo "user opt: -e"${LOW_OPTARG}
        if [[ "$OPTARG" != "" ]]; then
          MS_BACKEND=$OPTARG
        fi
        ;;
      V)
        echo "user opt: -V"${LOW_OPTARG}
        if [[ "$OPTARG" != "" ]]; then
          MS_VERSION=$OPTARG
        fi
        ;;
      p)
        if [[ "$OPTARG"  != "" ]]; then
          MS_WHL_LIB_PATH=$OPTARG
        else
          echo "Invalid value ${LOW_OPTARG} for option -e"
          usage
          exit 1
        fi
        ;;
      d)
        echo "user opt: -d"${LOW_OPTARG}
        DEBUG_MODE="on"
        ;;
      j)
        echo "user opt: -j"${LOW_OPTARG}
        THREAD_NUM=$OPTARG
        ;;
      v)
        echo "user opt: -v"${LOW_OPTARG}
        VERBOSE="VERBOSE=1"
        ;;
      c)
        check_on_off $OPTARG c
        ENABLE_COVERAGE="$OPTARG"
        ;;
      a)
        check_on_off $OPTARG a
        ENABLE_ASAN="$OPTARG"
        ;;
      t)
        echo "user opt: -t"${LOW_OPTARG}
        RUN_TESTCASES="$OPTARG"
        ;;
      S)
        check_on_off $OPTARG S
        ENABLE_GITEE="$OPTARG"
        echo "enable download from gitee"
        ;;
      *)
        echo "Unknown option ${opt}!"
        usage
        exit 1
    esac
  done
}

checkopts "$@"
echo "---------------- MindSpore Serving: build start ----------------"
mkdir -pv "${BUILD_PATH}/package/mindspore_serving/lib"
if [[ "$MS_BACKEND" != "" ]]; then
  git submodule update --init third_party/mindspore
fi

# Create building path
build_mindspore_serving()
{
  echo "start build mindspore_serving project."
  mkdir -pv "${BUILD_PATH}/mindspore_serving"
  cd "${BUILD_PATH}/mindspore_serving"
  CMAKE_ARGS="-DDEBUG_MODE=$DEBUG_MODE -DBUILD_PATH=$BUILD_PATH"
  CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_PYTHON=${ENABLE_PYTHON}"
  CMAKE_ARGS="${CMAKE_ARGS} -DTHREAD_NUM=${THREAD_NUM}"
  if [[ "X$ENABLE_COVERAGE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_COVERAGE=ON"
  fi
  if [[ "X$ENABLE_ASAN" = "Xon" ]]; then
      CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_ASAN=ON"
  fi
  if [[ "$MS_BACKEND" != "" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DMS_BACKEND=${MS_BACKEND}"
  fi
  if [[ "$MS_WHL_LIB_PATH" != "" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DMS_WHL_LIB_PATH=${MS_WHL_LIB_PATH}"
  fi
  if [[ "$MS_VERSION" != "" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DMS_VERSION=${MS_VERSION}"
  fi
  if [[ "X$RUN_TESTCASES" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_TESTCASES=ON"
  fi
  if [[ "X$ENABLE_GITEE" = "Xon" ]]; then
    CMAKE_ARGS="${CMAKE_ARGS} -DENABLE_GITEE=ON"
  fi
  echo "${CMAKE_ARGS}"
  cmake ${CMAKE_ARGS} ../..
  if [[ -n "$VERBOSE" ]]; then
    CMAKE_VERBOSE="--verbose"
  fi
  cmake --build . --target package ${CMAKE_VERBOSE} -j$THREAD_NUM
  echo "success building mindspore_serving project!"
}

build_mindspore_serving

echo "---------------- mindspore_serving: build end   ----------------"
