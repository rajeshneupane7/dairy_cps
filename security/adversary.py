import numpy as np

class Adversary:
    def inject_fdi_attack(self, data, attack_type="step"):
        """Corrupts Rumination readings while keeping Milk readings normal."""
        hacked_data = data.copy()
        
        if attack_type == "step":
            # Sudden 150-minute drop in reported rumination
            hacked_data['RT'] -= 150
            
        elif attack_type == "ramp":
            # Gradual decrease to mimic subclinical illness
            hacked_data['RT'] -= np.linspace(0, 100, len(data))
            
        elif attack_type == "replay":
            # Replaces current data with 'normal' data to hide a real sickness
            hacked_data['RT'] = 520 # Fake 'healthy' constant
            
        return hacked_data