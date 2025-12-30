from models.physical_twin import PhysicalTwin
from security.adversary import Adversary
from decision_logic import classify_event

# 1. Initialization
cow_402 = PhysicalTwin(cow_id=402)
# Load your 2-year CSV and train the model
# cow_402.fit_parameters(my_2_year_data)

# 2. Daily Simulation Loop
for day in range(305):
    # Get actual physical readings (Ground Truth)
    real_data = {"Milk": 32, "RT": 510, "THI": 68}
    
    # Adversary injects something into the Cyber stream
    cyber_report = Adversary().inject_fdi_attack(real_data, "step")
    
    # 3. Detection Engine
    expected = cow_402.predict_expected(day, cyber_report['THI'])
    
    # Check: Is the Milk vs RT relationship physically possible?
    # If Reported RT is low, but Reported Milk is high -> Correlation is broken
    correlation_broken = abs(cyber_report['RT'] - expected['RT']) > 100
    
    # 4. Final Classification
    result = classify_event(individual_anomaly=True, 
                            pen_anomaly=False, 
                            correlation_broken=correlation_broken)
    
    print(f"Day {day}: {result}")
