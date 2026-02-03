# Project Overview

This is a Python-based simulation of a dairy Cyber-Physical System (CPS). The project models the physical characteristics of a dairy cow, such as milk production and rumination time, to detect anomalies. The system is designed to distinguish between biological events (e.g., illness) and cyber-attacks (e.g., sensor manipulation).

The core of the project is a 305-day simulation for a single cow (`cow_402`). In each daily loop, the system simulates:
1.  **Data Generation:** Ground truth data for milk production and rumination time is generated.
2.  **Adversarial Attack:** An adversary injects false data into the system.
3.  **Anomaly Detection:** The system's detection engine compares the reported data against a predictive model of the cow's expected biological behavior.
4.  **Event Classification:** Based on the anomaly detection, the system classifies the event as normal, a biological issue (e.g., mastitis), a pen-level event (e.g., diet change), or a cyber-attack.

## Key Technologies

*   **Python:** The entire project is written in Python.
*   **Scientific Computing:** `numpy` and `scipy` are used for numerical operations and scientific modeling, specifically `curve_fit` for fitting the Wood's model parameters.
*   **Machine Learning:** `scikit-learn` is used for anomaly detection with an `IsolationForest` model.

## Building and Running

There is no formal build process for this project. To run the simulation, you need to have Python and the required libraries installed.

1.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the main simulation:**
    ```bash
    python main.py
    ```

## Development Conventions

*   **Modular Structure:** The code is organized into modules with clear separation of concerns:
    *   `main.py`: Main application entry point.
    *   `decision_logic.py`: Core classification logic.
    *   `models/`: Data models and physical simulations.
    *   `security/`: Adversarial and anomaly detection logic.
*   **Modeling:** The `PhysicalTwin` class uses the Wood's model, a common approach in dairy science for modeling lactation curves.
*   **Anomaly Detection:** A multi-faceted approach to anomaly detection is used, combining:
    *   **Model-based detection:** Comparing sensor readings to the predictions of the `PhysicalTwin` model.
    *   **Statistical detection:** Using Z-scores to identify outliers compared to pen-level data.
    *   **Machine learning-based detection:** Employing an `IsolationForest` for unsupervised anomaly detection.
