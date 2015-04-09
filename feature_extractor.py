import numpy as np
from sklearn.gaussian_process import GaussianProcess

def squared_exponential_periodic_1D(theta, d):
    theta = np.asarray(theta, dtype=np.float)
    d = np.asarray(d, dtype=np.float)
    return np.exp(-theta[0] * np.sum(np.sin(abs(d)) ** 2, axis=1))

def fold_time_series(time_point, period, div_period):
    real_period = period / div_period
    return time_point % real_period  # modulo real_period


def unfold_sample(x, color):
    """Operates inplace"""
    real_period = x['period'] / x['div_period']
    phase = (x['time_points_%s' % color] % real_period) / real_period * 2 * np.pi
    order = np.argsort(phase)
    x['phase_%s' % color] = phase[order]
    x['light_points_%s' % color] = np.array(x['light_points_%s' % color])[order]
    x['error_points_%s' % color] = np.array(x['error_points_%s' % color])[order]
    x['time_points_%s' % color] = np.array(x['time_points_%s' % color])[order]

class FeatureExtractor(object):

    def __init__(self):
        pass

    def fit(self, X_dict, y):
        pass

    def transform(self, X_dict):
        num_points_per_period = 200
        bins_per_period = 10
        sampling_rate = num_points_per_period / bins_per_period
        t_test = np.linspace(-2 * np.pi, 4 * np.pi, 3 * num_points_per_period)

        X = []
        for x in X_dict:
            real_period = x['period'] / x['div_period']
            x_new = [x['magnitude_b'], x['magnitude_r'], real_period]
            for color in ['r', 'b']:
                unfold_sample(x, color=color)
                x_train = x['phase_' + color]
                y_train = x['light_points_' + color]
                y_sigma = x['error_points_' + color]
                try:
                    gp = GaussianProcess(regr='constant', theta0=1./1.0, thetaL=1./50., thetaU=1./0.1, 
                                 corr=squared_exponential_periodic_1D,
                                 nugget=y_sigma*y_sigma)
                    gp.fit(x_train[:, np.newaxis], y_train[:, np.newaxis])
                except (Exception, ValueError):
                    # sklearn GP bug: x's cannot be identical even if there are errors 
                    x_train, unique_indexes = np.unique(x_train, return_index=True)
                    y_train = y_train[unique_indexes]
                    y_sigma = y_sigma[unique_indexes]
                    gp = GaussianProcess(regr='constant', theta0=1./1., thetaL=1./50., thetaU=1./0.1, 
                                 corr=squared_exponential_periodic_1D,
                                 nugget=y_sigma*y_sigma)
                    gp.fit(x_train[:, np.newaxis], y_train[:, np.newaxis])
                # this is the function you should play with, three periods (although for
                # now it is a bit longer than 3 periods, not sure why)
                y_test = gp.predict(t_test[:, np.newaxis])[:,0]
                length_scale = np.sqrt(2/gp.theta_[0][0])
                x_new.append(length_scale)
                miny = min(y_test)
                amplitude = max(y_test) - miny
                x_new.append(amplitude)
                # first max after t = 0
                amax_index = num_points_per_period + np.argmax(y_test[num_points_per_period:2 * num_points_per_period])
                # sample points from second period [0, 2pi]
                gp_samples = y_test[amax_index : amax_index + num_points_per_period : sampling_rate]
                # normalize sampled points between 0 and 1
                gp_samples_normalized = 1 - (gp_samples - miny) / amplitude
                for gp_sample in gp_samples_normalized:
                    x_new.append(gp_sample)

            X.append(x_new)
            
        return np.array(X) 