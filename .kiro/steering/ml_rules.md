# Steering Rules

1. Always avoid zero division in financial ratios using `1e-5`.
2. All random state seeds must be pinned to 42 for reproducible evaluations.
3. Separation of Concerns: No training logic inside Streamlit UI code.