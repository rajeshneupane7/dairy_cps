import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

class AnomalyEngine:
    def __init__(self, sensitivity=3.0):
        self.sensitivity = sensitivity # Standard deviations for Z-score

    def get_z_score_anomaly(self, cow_data, pen_data):
        """
        Detects if a cow is an outlier compared to its Pen-ID peers.
        """
        pen_mean = pen_data['Rumination_Time'].mean()
        pen_std = pen_data['Rumination_Time'].std()
        
        z_score = (cow_data['Rumination_Time'] - pen_mean) / (pen_std + 1e-6)
        return np.abs(z_score) > self.sensitivity

    def get_physical_residual(self, cow_data, model):
        """
        Physical-Based Anomaly: Does the reported Rumination match 
        what the model predicts based on the current Milk Yield and THI?
        """
        # Model predicts RT = alpha + beta1*Milk + beta2*THI
        predicted_rt = model.predict(cow_data[['Milk_Yield', 'THI']])
        residual = np.abs(cow_data['Rumination_Time'] - predicted_rt)
        
        # If residual is high, the 'Cyber' signal is lying about the 'Physical' state
        return residual > (predicted_rt * 0.15) # 15% deviation threshold

    def machine_learning_check(self, features):
        """
        Uses an Isolation Forest to detect 'Unseen' patterns 
        in the multi-dimensional data (Milk, RT, WI, THI).
        """
        clf = IsolationForest(contamination=0.01, random_state=42)
        preds = clf.fit_predict(features) # -1 is anomaly, 1 is normal
        return preds