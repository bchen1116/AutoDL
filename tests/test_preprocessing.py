import sys
sys.path.append('../')
import numpy as np
from AutoDL.preprocessing import channels_to_first, normalize

def test_channels_to_first_no_channel():
    no_channel = np.array([[1, 2, 3, 4],
                           [2, 3, 4, 5],
                           [3, 4, 5, 6],
                           [1, 2, 3, 5]])
    result = channels_to_first(no_channel)
    expected_result = np.stack((no_channel, no_channel, no_channel), axis=0)
    assert result.shape == (3, 4, 4)
    np.testing.assert_equal(result, expected_result)


def test_channels_to_first_one_channel():
    one_channel = np.array([[[1, 2, 3, 4],
                            [2, 3, 4, 5],
                            [3, 4, 5, 6],
                            [1, 2, 3, 5]]])
    result = channels_to_first(one_channel)
    no_channel = one_channel.reshape(4, 4)
    expected_result = np.stack((no_channel, no_channel, no_channel), axis=0)
    assert result.shape == (3, 4, 4)
    np.testing.assert_equal(result, expected_result)


def test_channels_to_first_three_channels():
    s = [[1, 2, 3, 4],
         [2, 3, 4, 5],
         [3, 4, 5, 6],
         [1, 2, 3, 5]]
    channels = np.array([s, s, s])
    result = channels_to_first(channels)
    assert result.shape == (3, 4, 4)
    np.testing.assert_equal(result, channels)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def test_normalize_bounded():
    s = [[0.8, 0.1, 0.3, 0.5],
         [0.4, 0.5, 0.6, 0.3]]
    input_matrix = np.array([s, s, s])
    expected_result = input_matrix
    for i in range(3):
        expected_result[i, :, :] -= mean[i]
        expected_result[i, :, :] /= std[i]
    result = normalize(input_matrix)
    np.testing.assert_equal(result, expected_result)


def test_normalize_unbounded():
    s = [[1, 100, 200, 255],
         [120, 160, 180, 240]]
    input_matrix = np.array([s, s, s])
    expected_result = np.divide(input_matrix, 255.)
    for i in range(3):
        expected_result[i, :, :] -= mean[i]
        expected_result[i, :, :] /= std[i]
    result = normalize(input_matrix)
    np.testing.assert_equal(result, expected_result)
