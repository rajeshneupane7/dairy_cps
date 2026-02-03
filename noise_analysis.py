import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import warnings

# Suppress technical warnings for cleaner logs
warnings.filterwarnings("ignore")

# ==========================================
# 1. PHYSICAL TWIN CLASS
# ==========================================
class PhysicalTwin:
    def __init__(self, cow_id):
        self.cow_id = cow_id
        self.milk_params = None
        self.models = {
            'milk_refiner': LinearRegression(),
            'rt_model': LinearRegression(),
            'wi_model': LinearRegression()
        }
        self.sigmas = {'milk': 0, 'rt': 0, 'wi': 0}
        self.is_trained = False

    def woods_model(self, t, a, b, c):
        """Biological foundation for lactation yield."""
        return a * (t**b) * np.exp(-c * t)

    def _prepare_X(self, primary_val, thi):
        """Creates feature matrix: [Value, THI, THI^2]"""
        return np.array([[primary_val, thi, thi**2]])

    def train(self, history):
        """Learns biological baselines and variance from clean historical data."""
        # Filter for days without health or management events
        clean_data = history[history['events'].isna()].copy()
        if len(clean_data) < 40:
            return

        try:
            # A. Fit Wood's Model (Yield Foundation)
            popt, _ = curve_fit(self.woods_model, clean_data['DIM'], clean_data['Milk'], p0=[15, 0.2, 0.004])
            self.milk_params = popt
            
            # B. Train Multivariate Milk Refiner (Corrects Wood's for THI)
            woods_preds = self.woods_model(clean_data['DIM'], *self.milk_params)
            X_m = pd.DataFrame({'w': woods_preds, 't': clean_data['THI'], 'ts': clean_data['THI']**2})
            self.models['milk_refiner'].fit(X_m, clean_data['Milk'])
            self.sigmas['milk'] = np.sqrt(mean_squared_error(clean_data['Milk'], self.models['milk_refiner'].predict(X_m)))

            # C. Train RT Model (Rumination ~ f(Milk, THI))
            X_env = pd.DataFrame({'m': clean_data['Milk'], 't': clean_data['THI'], 'ts': clean_data['THI']**2})
            self.models['rt_model'].fit(X_env, clean_data['RT'])
            self.sigmas['rt'] = np.sqrt(mean_squared_error(clean_data['RT'], self.models['rt_model'].predict(X_env)))

            # D. Train WI Model (Water ~ f(Milk, THI))
            self.models['wi_model'].fit(X_env, clean_data['Water'])
            self.sigmas['wi'] = np.sqrt(mean_squared_error(clean_data['Water'], self.models['wi_model'].predict(X_env)))

            self.is_trained = True
        except Exception as e:
            pass # Handle specific logging if needed

    def audit_record(self, row, z_threshold=3.0):
        """Checks a single day's data against learned biological truth."""
        if not self.is_trained:
            return {"status": "INSUFFICIENT_DATA", "z_score": 0}

        # 1. Predict what the cow 'Should' be doing
        base_milk = self.woods_model(row['DIM'], *self.milk_params)
        p_milk = self.models['milk_refiner'].predict(self._prepare_X(base_milk, row['THI']))[0]
        p_rt = self.models['rt_model'].predict(self._prepare_X(row['Milk'], row['THI']))[0]
        p_wi = self.models['wi_model'].predict(self._prepare_X(row['Milk'], row['THI']))[0]

        # 2. Calculate Z-Scores (Standard Deviations from Normal)
        z_milk = abs(row['Milk'] - p_milk) / self.sigmas['milk']
        z_rt = abs(row['RT'] - p_rt) / self.sigmas['rt']
        z_wi = abs(row['Water'] - p_wi) / self.sigmas['wi']

        # 3. Decision Logic
        max_z = max(z_milk, z_rt, z_wi)
        is_anomaly = max_z > z_threshold
        has_event = pd.notna(row['events'])

        if is_anomaly and not has_event:
            status = "CYBER_ANOMALY"
        elif is_anomaly and has_event:
            status = "BIOLOGICAL_EVENT"
        else:
            status = "NORMAL"

        return {"status": status, "z_score": max_z, "pred_milk": p_milk}

# ==========================================
# 2. SYSTEM COORDINATOR
# ==========================================
class DairyDigitalTwinSystem:
    def __init__(self, data):
        """
        Parameters
        ----------
        data : str or pandas.DataFrame
            Path to CSV file or a preloaded DataFrame
        """
        if isinstance(data, pd.DataFrame):
            self.df = data.copy()
        elif isinstance(data, str):
            self.df = pd.read_csv(data)
        else:
            raise ValueError("data must be a pandas DataFrame or a file path (str)")

        self.twins = {}
        self.results = []

    def run_pipeline(self):
        print(f"--- Starting Analysis on {self.df['cow_id'].nunique()} cows ---")
        
        # Process each cow
        for cow_id, group in self.df.groupby('cow_id'):
            # Split: 80% train (history), 20% test (current)
            split = int(len(group) * 0.8)
            train_df = group.iloc[:split]
            test_df = group.iloc[split:]
            
            twin = PhysicalTwin(cow_id)
            twin.train(train_df)
            self.twins[cow_id] = twin
            
            # Audit the test period
            for _, row in test_df.iterrows():
                audit = twin.audit_record(row)
                self.results.append({
                    'date': row['date'],
                    'cow_id': cow_id,
                    'sensor_group': row['sensor_group'],
                    **audit
                })

    def generate_report(self):
        report_df = pd.DataFrame(self.results)
        
        # 1. Detect System-Level Cyber Attack (Temporal)
        daily_cyber = report_df[report_df['status'] == "CYBER_ANOMALY"].groupby('date').size()
        attack_dates = daily_cyber[daily_cyber > (self.df['cow_id'].nunique() * 0.10)]
        
        # 2. Detect Infrastructure Breach (Spatial)
        group_stats = report_df.groupby('sensor_group').agg(
            total=('cow_id', 'nunique'),
            anomalies=('status', lambda x: (x == 'CYBER_ANOMALY').sum())
        )
        group_stats['breach_prob'] = group_stats['anomalies'] / group_stats['total']
        targeted_groups = group_stats[group_stats['breach_prob'] > 0.30]

        # Output Results
        print("=== SYSTEM AUDIT REPORT ===")
        print(f"Unexplained Anomalies Found: {len(report_df[report_df['status'] == 'CYBER_ANOMALY'])}")
        
        if not attack_dates.empty:
            print(f"[!] SYSTEM-WIDE ATTACK DETECTED ON DATES: {attack_dates.index.tolist()}")
            
        if not targeted_groups.empty:
            print(f"[!] INFRASTRUCTURE BREACH SUSPECTED IN GROUPS: {targeted_groups.index.tolist()}")
            
        return report_df

data_dir='/home/rajesh/work/data/data_move_-1.csv'
df= pd.read_csv(data_dir)
selected_df=df[['date_x', 'Animal_ID', 'Event', 'THI', 'yield', 'water_intake in l', 'Days_in_Milk', 'rum_index', 'Group_ID']]
selected_df=selected_df.rename(columns={
    'date_x': 'date',
    'Animal_ID': 'cow_id',
    'Days_in_Milk': 'DIM',
    'yield': 'Milk',
    'rum_index': 'RT',
    'water_intake in l': 'Water',
    'THI': 'THI',
    'Event': 'events',
    'Group_ID': 'sensor_group'
})

def inject_noise(df, noise_level):
    noisy_df = df.copy()
    # Add noise to the 'RT' column of the test data part
    mean_rt = noisy_df['RT'].mean()
    noise = np.random.normal(0, mean_rt * noise_level, noisy_df.shape[0])
    noisy_df['RT'] += noise
    return noisy_df

noise_levels = [0.01, 0.05, 0.1, 0.15, 0.2]
results_summary = {}

if selected_df is not None:
    for noise in noise_levels:
        print(f"--- Running analysis with {noise*100:.2f}% noise ---")
        noisy_df = inject_noise(selected_df, noise)
        
        # Rerun the system with noisy data
        system_noisy = DairyDigitalTwinSystem(noisy_df)
        system_noisy.run_pipeline()
        results_noisy = system_noisy.generate_report()
        
        num_anomalies = len(results_noisy[results_noisy['status'] == 'CYBER_ANOMALY'])
        results_summary[noise] = num_anomalies
        print(f"Found {num_anomalies} anomalies with {noise*100:.2f}% noise")
        print("\n" + "=" * 40 + "\n")

    print("--- Noise Injection Summary ---")
    for noise, anomalies in results_summary.items():
        print(f"Noise Level: {noise*100:.2f}% -> Detected Anomalies: {anomalies}")
else:
    print("Please load a dataframe into 'selected_df' to run the noise analysis.")
