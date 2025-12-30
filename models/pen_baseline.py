import pandas as pd

class PenBaseline:
    def __init__(self, pen_id):
        self.pen_id = pen_id

    def get_pen_metrics(self, current_herd_data):
        """Calculates the mean and standard deviation for the entire pen."""
        pen_data = current_herd_data[current_herd_data['Pen_ID'] == self.pen_id]
        return {
            'avg_milk': pen_data['Milk'].mean(),
            'avg_rt': pen_data['RT'].mean(),
            'std_rt': pen_data['RT'].std()
        }

    def check_for_diet_shift(self, pen_metrics, historical_pen_avg):
        """If the entire pen drops RT/Milk simultaneously, it is a Diet Shift."""
        threshold = 0.10 # 10% drop
        if pen_metrics['avg_rt'] < (historical_pen_avg * (1 - threshold)):
            return True # This is a pen-level management event
        return False