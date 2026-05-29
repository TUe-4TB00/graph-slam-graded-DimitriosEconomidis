import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # robot moves at 45 degrees for 2 meters, then rotates to face up (total 90 deg rotation)
    dx = math.sqrt(2)
    dy = math.sqrt(2)
    dtheta = math.pi / 2

    # TODO: Add the odometry factor between X(3) and X(4) to the graph (BetweenFactorPose2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), gtsam.Pose2(dx, dy, dtheta), ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(4) and add it to the graph
    pose3 = initial_estimate.atPose2(X(3))
    x4 = pose3.x() + dx
    y4 = pose3.y() + dy
    theta4 = pose3.theta() + dtheta
    initial_estimate.insert(X(4), gtsam.Pose2(x4, y4, theta4))

    return graph, initial_estimate