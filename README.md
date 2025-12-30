# CryptoPrice-Forecasting

## 1. Project Overview

This project is a **course capstone assignment** that applies **Machine Learning and Deep Learning models**
to **cryptocurrency price forecasting**.

Three different modeling approaches are explored.  
Detailed implementations, experiments, and accompanying papers are provided in the `notebook/` directory.

---

2. Environment Setup

To reproduce the experiments, follow these steps:

2.1 Clone the Repository
git clone <repository-url>
cd CryptoPrice-Forecasting

2.2 Create a Virtual Environment
python -m venv venv


Activate the environment:

Windows

venv\Scripts\activate


Linux / macOS

source venv/bin/activate

2.3 Install Dependencies
pip install -r requirements.txt

3. Experiments and Training

All experiments are implemented as Jupyter Notebooks in the notebook/ directory.

Each notebook includes:

Data preprocessing and normalization

Model training and evaluation

Result visualization

4. Saving Trained Models

After training the deep learning model (e.g., RNN or LSTM), save the trained model weights:

torch.save(rnn_model.state_dict(), "rnn_model_weights.pth")

5. Saving the Scaler (Important)

The scaler used during training must be saved to ensure correct preprocessing during inference:

import joblib
joblib.dump(scaler, "scaler.pkl")


Note: The Streamlit demo requires this scaler to transform new input data consistently.

6. Running the Streamlit Demo
6.1 Navigate to the Demo Folder
cd demo

6.2 Update Paths in app.py

Before running the demo, make sure to update the paths to the trained model and dataset.

MODEL_PATH = "../rnn_model_weights.pth"
DATA_PATH = "../data/crypto_price_2025.csv"

6.3 Run the Demo Application
streamlit run app.py

7. Notes

This project is intended for educational and research purposes only.

Forecasting results should not be considered financial advice.

8. References

Relevant academic papers are provided in the notebook/paper/ directory.
