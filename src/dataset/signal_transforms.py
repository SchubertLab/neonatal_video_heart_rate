import numpy as np


class RandomGaussianNoise(object):

    def __init__(self, median_noise=0, std_noise_factor=0.1):
        self.median_noise = median_noise
        self.std_noise_factor = std_noise_factor

    def __call__(self, input_signal):
        std_noise = self.std_noise_factor * np.std(input_signal)
        noise = np.random.normal(self.median_noise, std_noise, input_signal.shape)
        signal_out = input_signal + noise
        return signal_out


class RandomSignalFlip(object):

    def __init__(self, int_options=5):
        self.int_options = int_options

    def __call__(self, input_signal):
        int_rd = np.random.randint(1, self.int_options)
        vector_flipped = input_signal
        if int_rd == 1:
            vector_flipped = np.flipud(vector_flipped)
        if int_rd == 2:
            vector_flipped = np.fliplr(vector_flipped)
        if int_rd == 3:
            vector_flipped = np.fliplr(vector_flipped)
            vector_flipped = np.flipud(vector_flipped)
        return vector_flipped


class RandomSlopeNoise(object):

    def __init__(self, slope_range=[-1.0, 1.0], interc_range=[-1.0, 1.0]):
        self.slope_range = slope_range
        self.interc_range = interc_range

    def __call__(self, input_signal):
        slope = np.random.uniform(
            low=self.slope_range[0],
            high=self.slope_range[1],
        )
        intercept = np.random.uniform(
            low=self.interc_range[0],
            high=self.interc_range[1],
        )
        slope_signal = slope * input_signal + intercept
        return slope_signal


class RandomStepNoise(object):

    def __init__(self, factor_step_low=0.1, factor_step_high=1.0):
        self.factor_step_low = factor_step_low
        self.factor_step_high = factor_step_high

    def __call__(self, input_signal):
        factor_step = np.random.uniform(
            low=self.factor_step_low,
            high=self.factor_step_high,
        )
        step_a = np.random.uniform(low=0, high=1.0)
        step_b = np.random.uniform(low=0, high=1.0)
        signal_size = input_signal.shape[0]

        if step_b > step_a:
            step_start = int(step_a * signal_size)
            step_end = int(step_b * signal_size)
        else:
            step_start = int(step_b * signal_size)
            step_end = int(step_a * signal_size)

        noise_step_signal = np.zeros(input_signal.shape)
        noise_step_signal[step_start:step_end, :, :] = np.ones(input_signal.shape)[step_start:step_end, :, :] * np.std(
            input_signal) * factor_step

        step_signal = input_signal + noise_step_signal
        return step_signal


class MinMaxNorm(object):

    def __init__(self, scaling=0):
        self.scaling = scaling

    def __call__(self, input_signal):
        output_signal_norm = np.ones(input_signal.shape)
        for dim_1 in range(input_signal.shape[1]):
            for dim_2 in range(input_signal.shape[2]):
                temp_signal = input_signal[:, dim_1, dim_2]
                a_min = np.min(temp_signal)
                a_max = np.max(temp_signal)
                diff = a_max - a_min
                array_out = temp_signal - a_min
                if diff != 0:
                    array_out = np.divide(array_out, diff)
                output_signal_norm[:, dim_1, dim_2] = array_out
        return output_signal_norm
