import numpy as np

class Kalman:
    def __init__(self, state_dim: int = 6, obs_dim: int = 2):
        self.state_dim      = state_dim
        self.obs_dim        = obs_dim
        self._offset        = np.zeros((obs_dim))
        
        self.noise          = np.matrix(np.eye(state_dim)*1e-4)
        self.observation    = np.matrix(np.eye(obs_dim)*0.01)
        self.transition     = np.matrix(np.eye(state_dim))
        self.measurement    = np.matrix(np.zeros((obs_dim, state_dim)))
        self.gain           = np.matrix(np.zeros_like(self.measurement.T))
        self.covariance     = np.matrix(np.zeros_like(self.transition))
        self.state          = np.matrix(np.zeros((state_dim, 1)))
    
        if obs_dim == int(state_dim/3):
            # x( t + 1 ) = x( t ) + v( t ) + a( t ) / 2
            idx = np.r_[0:obs_dim]
            self.measurement[np.ix_(idx,idx)]                   = np.eye(obs_dim)
            self.transition                                     = np.eye(state_dim)
            self.transition[np.ix_(idx,idx+obs_dim)]            += np.eye(obs_dim)
            self.transition[np.ix_(idx,idx+obs_dim*2)]          += 0.5 * np.eye(obs_dim)
            self.transition[np.ix_(idx+obs_dim,idx+obs_dim*2)]  += np.eye(obs_dim)
    
    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value: np.ndarray):
        self._offset = np.array([value]).T

    def update(self, value: np.ndarray):
        pos = value - np.squeeze(self.offset)
        if pos.ndim == 1: pos = np.matrix(pos).T
        
        # make prediction
        self.state      = self.transition * self.state
        self.covariance = self.transition * self.covariance * self.transition.T + self.noise
        
        # compute optimal kalman gain factor
        self.gain       = self.covariance * self.measurement.T * np.linalg.inv(self.measurement * self.covariance * self.measurement.T + self.observation)
        
        # correction based on observation
        self.state      = self.state + self.gain * (pos - self.measurement * self.state)
        self.covariance = self.covariance - self.gain * self.measurement * self.covariance

    def predict(self) -> np.ndarray:
        return np.asarray(self.measurement * self.state) + self.offset
