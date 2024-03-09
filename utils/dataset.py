import os

import numpy as np
import gpytorch.kernels
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from utils.utils import get_bigmij, get_smallmij, get_delta, get_cone_params


### DISCRETE DATASETS ###

DATASET_SIZES = {
    "SNW": 206,
    "Brake": 128,
    "PK2": 500,
    "VehicleCrashworthiness2K": 2000,
    "VehicleCrashworthiness1H": 100,
    "Marine": 500,
}

class Dataset:
    def __init__(self, cone_degree):
        input_scaler = MinMaxScaler()
        self.in_data = input_scaler.fit_transform(self.in_data)
        self.in_dim = len(self.in_data[0])

        output_scaler = StandardScaler(with_mean=True, with_std=True)
        self.out_data = output_scaler.fit_transform(self.out_data)
        self.out_dim = len(self.out_data[0])

        self.cone_degree = cone_degree
        self.W, self.alpha_vec, _ = get_cone_params(self.cone_degree, dim=self.out_dim)

        self.pareto_indices = None
        self.pareto = None
        self.delta = None

        # self.delta = get_delta(self.out_data, self.W, self.alpha_vec)
    
    def set_pareto_indices(self):
        self.find_pareto()
        print(f"For cone degree {self.cone_degree}, the pareto set indices are:")
        print(self.pareto_indices)

    def find_pareto(self):
        """
        Find the indices of Pareto designs (rows of out_data)
        :param mu: An (n_points, D) array
        :param W: (n_constraint,D) ndarray
        :param alpha_vec: (n_constraint,1) ndarray of alphas of W
        :return: An array of indices of pareto-efficient points.
        """
        out_data = self.out_data.copy()
        
        n_points = out_data.shape[0]
        is_efficient = np.arange(out_data.shape[0])
        
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(out_data):
            nondominated_point_mask = np.zeros(out_data.shape[0], dtype=bool)
            vj = out_data[next_point_index].reshape(-1,1)
            for i in range(len(out_data)):
                vi = out_data[i].reshape(-1,1)
                nondominated_point_mask[i] = (
                    (get_smallmij(vi, vj, self.W, self.alpha_vec) == 0)
                    and (get_bigmij(vi, vj, self.W) > 0)
                )
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            out_data = out_data[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        
        self.pareto_indices = is_efficient

    def get_params(self):
        if self.delta is None:
            self.delta = get_delta(self.out_data, self.W, self.alpha_vec)
        return self.delta, self.pareto_indices

class SNW(Dataset):
    def __init__(self, cone_degree):
        datafile = os.path.join('data', 'snw.csv')
        designs = np.genfromtxt(datafile, delimiter=';')
        self.out_data = np.copy(designs[:,3:])
        self.out_data[:,0] = -self.out_data[:,0]
        self.in_data = np.copy(designs[:,:3])

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]
    
    def set_pareto_indices(self):
        if self.cone_degree not in [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = [
                2,   3,   4,   5,   6,   7,   8,  10,  11,  12,  14,  18,  27,  28, 29, 30, 32, 33,
                34,  36,  37,  38,  39,  40,  42,  43,  44,  45,  46,  49,  50,  59,  61,  63,  64,
                66,  67,  80,  81,  96, 112, 128, 153, 154, 155, 160, 161, 162, 167, 168, 174, 187
            ]
        elif self.cone_degree == 60:
            self.pareto_indices =  [
                2,   3,   4,   5,   6,   7,   8,  10,  11,  12,  14,  18,  27,  28,  29,  30,  32,  33,
                36,  37,  38,  39,  40,  42,  43,  44,  45,  46,  49,  61,  63,  64,  66,  80,  81, 128,
                153, 154, 160, 161, 162, 167, 168, 174, 187,
            ]
        elif self.cone_degree == 90:
            self.pareto_indices = [
                2,   3,   4,   5,   6,   7,   8,  10,  11,  12,  14,  28,  29, 30, 32, 38, 40,  42,
                43,  45,  63, 160, 161, 167, 168, 174
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = [  2,  4,  6,  7,  8, 10, 11, 12,  14, 29, 160, 167, 168, 174  ]
        elif self.cone_degree == 135:
            self.pareto_indices = [  2,   4,   6,   7,   8,  10,  12,  14, 160, 167  ]

class DiskBrake(Dataset):
    def __init__(self, cone_degree):
        data = np.load(os.path.join('data', 'diskbrake.npy'), allow_pickle=True)
        self.in_data = data[:, :4]
        self.out_data = data[:, 4:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        pareto_dict = {
            10: [
                4,   8,   9,  13,  14,  15,  16,  20,  22,  24,  25,  26,  27,  30,  31,  34,  38,
                43, 45,  46,  47,  48,  50,  53,  54,  55,  56,  62,  63,  70,  75,  76,  78,  81,
                82,  85, 86,  88,  94,  95,  98,  99, 102, 103, 106, 113, 115, 116, 118, 121, 126
            ],
            15: [
                14,  15,  24,  25,  26,  27,  30,  31,  34,  38,  43,  45,  46,  47,  48,  53,  54,
                56,  62,  70,  75,  76,  78,  81,  82,  85,  86,  88,  94,  95,  99, 102, 103, 106,
                116, 118, 121, 126
            ],
            20: [
                14,  15,  24,  25,  26,  27,  30,  31,  34,  38,  43,  45,  46,  47,  48,  53,  54,
                62, 70,  76,  78,  81,  82,  85,  86,  88,  94,  99, 102, 103, 106, 116, 118, 121,
                126
            ],
            25: [
                14,  15,  24,  25,  27,  31,  34,  38,  43,  45,  46,  47,  48,  53,  54,  62,  70,
                76, 78,  81,  82,  85,  86,  88,  94,  99, 102, 106, 116, 118, 121, 126
            ],
            30: [
                14,  15,  24,  25,  27,  34,  38,  45,  46,  47,  48,  53,  62,  70,  76,  78,  81,
                85, 86,  88,  94, 102, 106, 116, 118, 121, 126
            ],
            35: [
                15,  25,  27,  38,  45,  46,  47,  48,  53,  62,  70,  78,  81,  85,  88,  94, 102,
                106, 116, 118, 121, 126
            ],
            40: [
                15,  25,  27,  38,  45,  46,  47,  48,  53,  62,  70,  78,  81,  85,  88,  94, 106,
                116, 118, 121, 126
            ],
            45: [
                15, 25, 27, 38, 45, 46, 48, 53, 62, 70, 78, 81, 85, 88, 94, 106, 118, 121, 126
            ],
            50: [
                15,  25,  27,  38,  45,  46,  53,  62,  70,  78,  81,  85,  88,  94, 106, 118, 121,
                126
            ],
            55: [ 15,  25,  38,  45,  46,  53,  62,  70,  78,  81,  88, 106, 118, 121, 126,],
            60: [ 15, 25, 38, 45, 46, 53, 62, 70, 78, 88, 106, 118, 121, 126 ],
            65: [ 15,  25,  38,  45,  46,  53,  62,  70,  78,  88, 106, 118, 121, 126,],
            70: [ 15,  25,  38,  45,  46,  53,  62,  70,  78, 106, 118, 121, 126,],
            75: [ 15,  25,  38,  53,  62,  78, 106, 118, 121, 126,],
            80: [ 25,  53,  62,  78, 106, 118, 121, 126,],
            85: [ 25,  53,  62,  78, 118, 121, 126,],
            90: [ 25, 53, 62, 78, 118, 121, 126 ],
            95: [ 25,  62,  78, 118, 121, 126,],
            100: [ 25,  62,  78, 118, 121, 126,],
            105: [ 25,  62,  78, 118, 126,],
            110: [ 25,  62,  78, 118, 126,],
            115: [ 25,  62,  78, 118, 126,],
            120: [ 25,  62,  118, 126,],
            125: [ 62, 118, 126,],
            130: [ 62, 118, 126,],
            135: [ 62, 118, 126,],
            140: [62,],
            145: [62,],
            150: [62,],
            155: [62,],
            160: [62,],
            165: [62,],
            170: [62,],
        }
        if self.cone_degree not in pareto_dict:
            super().set_pareto_indices()
        else:
            self.pareto_indices = pareto_dict[self.cone_degree]

class PK2(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'PK2.npy'), allow_pickle=True
        )
        self.in_data = data[:, :3]
        self.out_data = data[:, 3:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [60, 90, 120, 135]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = [
                43,  52,  82, 104, 118, 138, 144, 163, 186, 192, 204, 220, 234, 244, 264, 288,
                322, 331, 347, 386, 388, 446, 450, 456, 494, 499
            ]
        elif self.cone_degree == 90:
            self.pareto_indices = [
                118, 138, 204, 234, 264, 288, 331, 347, 388, 450, 456, 494, 499
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = [264, 288, 331, 347, 450, 456,]
        elif self.cone_degree == 135:
            self.pareto_indices = [264, 456]

class VehicleCrashworthiness2K(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'VehicleCrashworthiness2K.npy'), allow_pickle=True
        )
        self.in_data = data[:, :5]
        self.out_data = data[:, 5:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [45, 90, 135]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = [
                2,   12,   26,   35,   54,   95,   99,  131,  146,  169,  184,  210,  222,  234,
                290,  331,  347,  374,  402,  419,  432,  435,  442,  482,  500,  502,  503,  523,
                562,  563,  590,  628,  643,  673,  690,  710,  720,  745,  755,  760,  778,  827,
                852,  874,  875,  894,  930,  936,  964,  978,  990, 1018, 1043, 1106, 1138, 1154,
                1188, 1215, 1219, 1235, 1244, 1266, 1275, 1280, 1306, 1318, 1347, 1354, 1371, 1395,
                1490, 1508, 1514, 1566, 1611, 1622, 1650, 1659, 1666, 1667, 1690, 1696, 1731, 1748,
                1750, 1763, 1818, 1826, 1875, 1890, 1902, 1915, 1938, 1940, 1971, 1999
            ]
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                12,   26,   95,   99,  222,  234,  290,  374,  402,  435,  502,
                523,  562,  563, 590,  643,  690,  710,  720,  875,  978, 1043, 1154,
                1219, 1280, 1318, 1354, 1395, 1490, 1508, 1514, 1611, 1650, 1690, 1696,
                1731, 1890, 1902, 1938, 1940, 1971, 1999,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = [ 643, 1154, 1731 ]

class VehicleCrashworthiness1H(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'VehicleCrashworthiness1H.npy'), allow_pickle=True
        )
        self.in_data = data[:, :5]
        self.out_data = data[:, 5:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [45, 90, 135]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = [
                3,  6,  9, 10, 11, 15, 16, 22, 23, 25, 32, 34, 35, 40, 43, 45, 50, 51,
                53, 59, 60, 63, 66, 75, 78, 79, 81, 83, 85, 90, 91, 99,
            ]
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                6, 10, 11, 15, 16, 22, 25, 32, 35, 43, 50, 51, 53, 59, 60, 66, 78, 81, 85, 90,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = [22, 50, 81, 90]

class Marine(Dataset):
    def __init__(self, cone_degree):
        data = np.load(
            os.path.join('data', 'marine.npy'), allow_pickle=True
        )
        self.in_data = data[:, :6]
        self.out_data = data[:, 6:]

        super().__init__(cone_degree)

        self.model_kernel = gpytorch.kernels.RBFKernel

        self.set_pareto_indices()
        self.pareto = self.in_data[self.pareto_indices]

    def set_pareto_indices(self):
        if self.cone_degree not in [90]:  # [45, 60, 90, 120, 135]:
            super().set_pareto_indices()
        elif self.cone_degree == 45:
            self.pareto_indices = []
        elif self.cone_degree == 60:
            self.pareto_indices = []
        elif self.cone_degree == 90:
            self.pareto_indices = [
                4,  11,  18,  35,  51,  55,  67,  74,  94,  98, 107, 120, 127, 139, 151,
                163, 179, 189, 218, 225, 230, 243, 250, 259, 307, 315, 322, 328, 342, 343,
                352, 362, 371, 374, 378, 387, 403, 405, 411, 427, 447, 475, 482, 491,
            ]
        elif self.cone_degree == 120:
            self.pareto_indices = []
        elif self.cone_degree == 135:
            self.pareto_indices = []
