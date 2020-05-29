from utils import geometric_median, normalize
import numpy as np
import scipy
from scipy import stats

class Outlator():
    def detect_robust(self, data):
        data = normalize(data)
        self.mu = self.calculate_mu(data)
        self.sigma = self.calculate_sigma(data)

        n = len(data)
        distances = []
        for i in range(n):
            centered = data[[i], :]-self.mu.T
            sigma_inv = np.linalg.inv(self.sigma)
            tmp = centered@sigma_inv@centered.T
            distances.append([float(tmp**(1/2)), i])

        threshold = scipy.stats.chi2.ppf(0.975, data.shape[1]) # chi squared p,0.975 quantile

        return [x[1] for x in distances if x[0] > threshold]

    def calculate_mu(self, data):
        n,p = data.shape
        e = np.ones((p, 1))
        
        mu_MM = geometric_median(data).reshape(-1, 1)
        v_mu = float((mu_MM.T@e)/p)

        # calcualte eta
        def calculate_A_B(X_i):
            norm = np.linalg.norm(X_i, 2)
            B = (X_i@X_i.T)/(norm**2)
            A = (1/norm)*(np.eye(p)-B)

            return A,B

        A,B = calculate_A_B(data[0, :].T)
        for i in range(1, n):
            A_i, B_i = calculate_A_B(data[[i], :].T)
            A = A+A_i
            B = B+B_i
        A_inv = np.linalg.inv(A)
        eta = np.trace((1/n)*(A_inv@B@A_inv))/np.linalg.norm(mu_MM-v_mu*e)**2

        return (1-eta)*mu_MM + eta*v_mu*e


    def calculate_sigma(self, data):
        n, p = data.shape
        e = np.ones((p,))

        # calcualte median and variance of median
        # mu_CCM = np.median(data, axis=0)
        mu_CCM = geometric_median(data)
        S_hat_CCM = 2.198*self.calculate_comedian(data, mu_CCM)

        # shrinkage for mu
        v_hat_mu = float((mu_CCM@e)/p)
        eta_hat = (((np.pi/(2*n))*np.trace(S_hat_CCM))/np.linalg.norm(mu_CCM-v_hat_mu*e, 2))
        mu_hat_sh_MM = (1-eta_hat)*mu_CCM + eta_hat*v_hat_mu*e
        
        # calculate sigma
        s_hat_sh_MM = 2.198*self.calculate_comedian(data, mu_hat_sh_MM)

        # shrinkage for sigma
        v_sigma = float(np.trace(S_hat_CCM)/p)
        A = s_hat_sh_MM - S_hat_CCM
        num = np.trace(A@A.T)/p
        B = s_hat_sh_MM - v_sigma*np.eye(p)
        den = np.trace(B@B.T)/p
        eta = num/den

        return (1-eta)*s_hat_sh_MM + eta*v_sigma*np.eye(p)

    def calculate_comedian(self, data, mu):
        p = data.shape[1]
        comedian = np.zeros((p,p), dtype=np.float32)

        for j in range(p):
            data_j = data[:, j] - mu[j]
            for t in range(p):
                data_t = data[:, t] - mu[t]
                result = np.multiply(data_j, data_t)
                comedian[j,t] = np.median(result)
        
        return comedian
    
    def detect_mahal(self,data):
        mu = np.mean(data, axis=0).reshape(-1,1)
        sigma = np.cov(data.T)
        sigma_inv = np.linalg.inv(sigma)
        distances = []
        for i in range(len(data)):
            cent = data[i]-mu.T
            distances.append([float(cent@sigma_inv@cent.T), i])

        threshold = scipy.stats.chi2.ppf(0.975, data.shape[1])
        return [x[1] for x in distances if x[0] > threshold]
    
    def detect_tuckey(self, data):
        u = np.random.normal(0,1,(300, data.shape[1]))
        scalar = data@u.T
        distances = []
        for i in range(len(data)):
            scalar2 = (data[i]@u.T).reshape(1, -1)
            replic = np.ones((data.shape[0], 1))@scalar2
            dif = scalar - replic
            difindicator = dif>0
            M = np.mean(difindicator, axis=0)
            distances.append([np.min(M), i])
        
        threshold = np.percentile([x[0] for x in distances], 5)
        return [x[1] for x in distances if x[0] < threshold]