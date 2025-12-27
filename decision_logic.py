def classify_event(individual_anomaly, pen_anomaly, correlation_broken):
    """
    Logic Table:
    1. Pen Anomaly (No) + Indiv Anomaly (No) -> NORMAL
    2. Pen Anomaly (Yes) -> DIET/MANAGEMENT CHANGE
    3. Pen Anomaly (No) + Indiv Anomaly (Yes) + Correlation OK -> HEALTH (Mastitis)
    4. Pen Anomaly (No) + Indiv Anomaly (Yes) + Correlation BROKEN -> CYBER ATTACK
    """
    if not individual_anomaly:
        return "STATUS: Normal"
    
    if not pen_anomaly and individual_anomaly:
        if correlation_broken:
            return "ALARM: CYBER ATTACK DETECTED (Sensor Manipulation)"
        else:
            return "ALARM: BIOLOGICAL EVENT (Mastitis/Illness)"
            
    if pen_anomaly:
        return "STATUS: Pen-Level Management/Diet Change"
        
    return "STATUS: Unknown Anomaly"