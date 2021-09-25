# x_k+1 | k
from data_fusion.utils.data import base_data
import numpy as np

from data_fusion.utils.helpers import reduce_measurement

"""
Initialise everything
"""

# init the "pick"-parameters

# initialise Cxx aka error covariance
Cxx_init = np.eye(4)

# initialise sigma
sigma = 1

# initialise Cvv aka R
R = np.eye(2)

# define / initialise Q aka process noise covariance aka Cww
q = np.array([[.125], [.125], [.5], [.5]])
Q = q @ ((sigma ** 2) * q.T)

# define / initialise F aka A aka T aka time update matrix
A = np.eye(4)
A[0][2] = 0.5
A[1][3] = 0.5

# define / initialise H aka the measurement matrix
H = np.eye(2, 4)

"""
Functions
"""


def predict_state_x(x):
    x_predict = A @ x
    return x_predict


# Cxx_k+1 | k
def predict_covariance_Cxx(current_Cxx):
    Cxx_predict = A @ current_Cxx @ A.T + Q
    return Cxx_predict


# calculate Kalman gain
def calc_kalman_gain(Cxx_predict):
    K_gain = Cxx_predict @ H.T @ np.linalg.inv(H @ Cxx_predict @ H.T + R)
    return K_gain


# calculate yk aka the measurements
def measurement_model_yk(x_current):
    y_k = H @ x_current
    return y_k


# x_k+1
def measurement_update_x(x_predicted, y_k, K_gain):
    x_measurement = x_predicted + K_gain @ (y_k - H @ x_predicted)
    return x_measurement


# Cxx_k+1
def measurement_covariance_Cxx(Cxx_predicted, K_gain):
    Cxx_measurement = Cxx_predicted - K_gain @ H @ Cxx_predicted
    return Cxx_measurement


def update_predictions_basic_kalman():
    x1 = base_data[0]['x']
    y1 = base_data[0]['y']
    x2 = base_data[1]['x']
    y2 = base_data[1]['y']

    x_init = [[x1], [y1], [x2 - x1], [y2 - y1]]

    c_meas = np.eye(4)  # Cxx_init
    x_meas = x_init

    collector = []

    for i in range(37):
        # Time Update
        x_meas = predict_state_x(x_meas)
        cxx_prediction = predict_covariance_Cxx(c_meas)

        # -> prediction

        # Measurement Update <- Measurement
        rm = reduce_measurement(i)
        gt_x, gt_y = rm['x'], rm['y']

        K = calc_kalman_gain(cxx_prediction)
        y_k = measurement_model_yk(x_meas)
        x_meas = measurement_update_x(x_meas, np.array([[gt_x], [gt_y]]), K)
        c_meas = measurement_covariance_Cxx(cxx_prediction, K)

        collector.append([x_meas[0][0], x_meas[1][0]])

    return np.asarray(collector, dtype=float)
