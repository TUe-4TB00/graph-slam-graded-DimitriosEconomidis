import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global`
    # which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function.
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # Initialize the optimizer
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Perform the optimization and print the result
    result = optimizer.optimize()
    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest sum of marginals.
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            # make independent copies so iterations don't interfere
            g = gtsam.NonlinearFactorGraph()
            for i in range(graph.size()):
                g.add(graph.at(i))
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            result = optimize(g, est)

            # TODO: Calculate marginal covariances for the relevant variables and visualize the updated factor graph with covariances
            marginals_obj = gtsam.Marginals(g, result)
            # The sum of the marginals for each landmark can be computed using marginals.marginalCovariance(L(x)).sum()
            total = marginals_obj.marginalCovariance(L(landmark)).sum()

            if total < best_sum:
                best_sum = total
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum

def minimize_errors(graph, initial_estimate, pose_options):
    #TODO: try different pose and landmark options here, and keep the one with the lowest resulting error.
    best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            # make independent copies so iterations don't interfere
            g = gtsam.NonlinearFactorGraph()
            for i in range(graph.size()):
                g.add(graph.at(i))
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            
            result = optimize(g, est)

            # Calculate marginal covariances
            marginals_obj = gtsam.Marginals(g, result)

            # TODO: create a list of errors (each index corresponds to a pose X1, X2, X3)
            list_of_errors = [
                marginals_obj.marginalCovariance(X(1)).sum(),
                marginals_obj.marginalCovariance(X(2)).sum(),
                marginals_obj.marginalCovariance(X(3)).sum()
            ]
            
            # TODO: compute the sum of the errors and return it along with the best pose and landmark
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum

#     best_pose = None
    best_landmark = None
    best_sum = float('inf')

    for pose_key, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            # make independent copies so iterations don't interfere
            g = gtsam.NonlinearFactorGraph()
            for i in range(graph.size()):
                g.add(graph.at(i))
            est = gtsam.Values(initial_estimate)

            g, est = add_pose(g, est, pose_5)
            result = optimize(g, est)
            g = add_landmark_measurement(g, result, pose_5, landmark)
            
            # CRITICAL: You must optimize again after adding the measurement!
            result = optimize(g, est) 

            marginals_obj = gtsam.Marginals(g, result)

            # Use the determinant (D-optimality) of the covariance matrices to measure "accuracy"
            list_of_errors = [
                np.linalg.det(marginals_obj.marginalCovariance(X(1))),
                np.linalg.det(marginals_obj.marginalCovariance(X(2))),
                np.linalg.det(marginals_obj.marginalCovariance(X(3)))
            ]
            
            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_sum:
                best_sum = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum