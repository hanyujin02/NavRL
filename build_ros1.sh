#!/bin/bash
# Workaround: /lib is a symlink to usr/lib on this system, causing cmake to compute
# wrong prefix paths. Explicitly pointing to /usr/lib paths fixes Boost and PCL discovery.
catkin_make \
  -DBoost_DIR=/usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0 \
  -DPCL_DIR=/usr/lib/x86_64-linux-gnu/cmake/pcl \
  "$@"
