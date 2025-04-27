import numpy as np
from scipy.stats import norm

class UniversalCUSUMDetector:
    def __init__(self, mode="adaptive", threshold=5.0, drift=0.01, min_distance=30,
                 alpha=0.02, beta=0.02, mean_diff=0.3):
        assert mode in ["adaptive", "hybrid"], "mode must be 'adaptive' or 'hybrid'"
        self.mode = mode
        self.threshold = threshold
        self.drift = drift
        self.min_distance = min_distance
        self.alpha = alpha
        self.beta = beta
        self.mean_diff = mean_diff
        self.reset()

    def reset(self):
        self.mean_sum = None
        self.mean_weights_sum = 0
        self.var_sum = None
        self.var_weights_sum = 0
        self.mean_estimate = 0
        self.var_estimate = 1
        self.stat = 0
        self.gp = 0
        self.gn = 0
        self.raw_cps = []

    def detect(self, series):
        data = series.values
        mean = np.mean(data)  # для классического CUSUM в режиме 'hybrid'

        for i, x in enumerate(data):
            if self.mode == "hybrid":
                # --- Классический CUSUM ---
                self.gp = max(0, self.gp + x - mean - self.drift)
                self.gn = min(0, self.gn + x - mean + self.drift)

                if self.gp > self.threshold or self.gn < -self.threshold:
                    self.raw_cps.append(i)
                    self.gp = 0
                    self.gn = 0

            # --- Адаптивный CUSUM ---
            if self.mean_sum is not None:
                self.mean_estimate = self.mean_sum / self.mean_weights_sum
                self.var_estimate = self.var_sum / self.var_weights_sum
                normalized = (x - self.mean_estimate) / np.sqrt(self.var_estimate)
                self.mean_sum = (1 - self.alpha) * self.mean_sum + x
                self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
                self.var_sum = (1 - self.beta) * self.var_sum + (normalized - self.mean_estimate) ** 2
                self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
            else:
                self.mean_sum = x
                self.mean_weights_sum = 1.0
                normalized = (x - self.mean_estimate) / np.sqrt(self.var_estimate)
                self.var_sum = (normalized - self.mean_estimate) ** 2
                self.var_weights_sum = 1.0

            zeta_k = np.log(norm.pdf(normalized, self.mean_diff, 1) / norm.pdf(normalized, 0., 1))
            self.stat = max(0, self.stat + zeta_k)

            if self.stat > self.threshold:
                self.raw_cps.append(i)
                self.stat = 0  # сброс адаптивной статистики

        # --- Фильтрация по min_distance ---
        filtered_cps = []
        last_cp = -self.min_distance
        for cp in sorted(set(self.raw_cps)):
            if cp - last_cp >= self.min_distance:
                filtered_cps.append(cp)
                last_cp = cp

        return filtered_cps
    
    def update(self, x):
        if self.mode == "hybrid":
            if self.mean_sum is None:
                self.mean_sum = x
                self.mean_weights_sum = 1.0
            mean = self.mean_sum / self.mean_weights_sum

            self.gp = max(0, self.gp + x - mean - self.drift)
            self.gn = min(0, self.gn + x - mean + self.drift)

            if self.gp > self.threshold or self.gn < -self.threshold:
                self.gp = 0
                self.gn = 0
                return True
            else:
                return False

        else:  # mode == 'adaptive'
            if self.mean_sum is not None:
                self.mean_estimate = self.mean_sum / self.mean_weights_sum
                self.var_estimate = self.var_sum / self.var_weights_sum
                normalized = (x - self.mean_estimate) / np.sqrt(self.var_estimate)
                self.mean_sum = (1 - self.alpha) * self.mean_sum + x
                self.mean_weights_sum = (1 - self.alpha) * self.mean_weights_sum + 1.0
                self.var_sum = (1 - self.beta) * self.var_sum + (normalized - self.mean_estimate) ** 2
                self.var_weights_sum = (1 - self.beta) * self.var_weights_sum + 1.0
            else:
                self.mean_sum = x
                self.mean_weights_sum = 1.0
                normalized = (x - self.mean_estimate) / np.sqrt(self.var_estimate)
                self.var_sum = (normalized - self.mean_estimate) ** 2
                self.var_weights_sum = 1.0

            zeta_k = np.log(norm.pdf(normalized, self.mean_diff, 1) / norm.pdf(normalized, 0., 1))
            self.stat = max(0, self.stat + zeta_k)

            if self.stat > self.threshold:
                self.stat = 0
                return True
            else:
                return False



