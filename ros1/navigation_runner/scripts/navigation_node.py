#!/usr/bin/env python3

import sys
# Strip ROS remapping args (__name:=, __log:=, etc.) before Hydra parses sys.argv
sys.argv = [a for a in sys.argv if not a.startswith('__')]

import rospy
from navigation import Navigation
import hydra
import os

FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts/cfg")
@hydra.main(config_path=FILE_PATH, config_name="deploy", version_base=None)
def main(cfg):
    rospy.init_node("navigation_node")
    nav = Navigation(cfg)
    nav.run()
    rospy.spin()


if __name__ == "__main__":
    main()