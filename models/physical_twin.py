import numpy as np
from scipy.optimize import curve_fit

class PhysicalTwin:
    def __init__(self, cow_id):
        self.cow_id = cow_id
        self.milk_params = None # [a, b, c] for Wood's Model
        self.rt_correlation = None # Correlation factor for Rumination vs Milk

    def woods_model(self, t, a, b, c):
        return a * (t**b) * np.exp(-c * t)

    def fit_parameters(self, historical_data):
        """Fits Wood's curve and RT correlations using 2-year history."""
        # Fit Milk Yield parameters
        popt, _ = curve_fit(self.woods_model, historical_data['DIM'], historical_data['Milk'])
        self.milk_params = popt
        
        # Fit Rumination correlation: RT = baseline + (coeff * Milk) - (coeff * THI)
        # Use simple linear regression or historical averages
        self.rt_ratio = (historical_data['RT'] / historical_data['Milk']).mean()

    def predict_expected(self, dim, thi):
        """Predicts what the cow SHOULD be doing biologically."""
        expected_milk = self.woods_model(dim, *self.milk_params)
        # Adjust for heat stress (THI)
        if thi > 72: expected_milk *= 0.95 
        
        expected_rt = expected_milk * self.rt_ratio
        return {"Milk": expected_milk, "RT": expected_rt}