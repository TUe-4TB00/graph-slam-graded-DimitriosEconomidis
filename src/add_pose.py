import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # robot rotates 45deg counter-clockwise, then 2m, then rotates 45deg.
    # total rotation: pi/2 and discplamenet at 45 deg: dx=sqrt(2), dy=sqrt(2)
    dx = math.sqrt(2)
    dy = math.sqrt(2)
    dtheta = math.pi / 2  #90 degrees total rotation

    # TODO: Add the odometry factor between X(3) and X(4) to the graph (BetweenFactorPose2)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), gtsam.Pose2(dx, dy, dtheta), ODOMETRY_NOISE))

    # TODO: Based on the odometry, find the initial estimate for the pose of X(4) and add it to the graph
    # X(3) is at (4, 0, 0); apply the relative motion to get global X(4)
    
    x4 = 4.0 + dx   
    y4 = 0.0 + dy  
    theta4 = dtheta  
    initial_estimate.insert(X(4), gtsam.Pose2(x4, y4, theta4))

    return graph, initial_estimate
