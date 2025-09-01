# Customer Churn Prediction App

This project is a web application for predicting customer churn using a machine learning model trained on the Telco Customer Churn dataset. The app is built with Streamlit and allows users to input customer details to predict the likelihood of churn.

## Features
- **Interactive Web App**: User-friendly interface for entering customer data and viewing predictions.
- **Machine Learning Model**: Random Forest Classifier trained on real-world telco data.
- **Visualization**: Probability of churn visualized with a bar chart.
- **Easy Deployment**: Run locally with minimal setup.

## Project Structure
```
app.py                  # Streamlit web application
train_model.py          # Script to train and save the churn prediction model
data/
    telco_churn.csv     # Dataset used for training
models/
    churn_model.pkl     # Trained model file
utils/                  # (Reserved for utility scripts)
```

## Getting Started

### 1. Clone the Repository
```bash
git clone <repo-url>
cd chrun_project
```

### 2. Install Dependencies
Make sure you have Python 3.7+ installed. Install required packages:
```bash
pip install streamlit pandas scikit-learn joblib matplotlib seaborn
```

### 3. Prepare the Data
Place the `telco_churn.csv` file in the `data/` directory. (Already included if you cloned the full repo.)

### 4. Train the Model (Optional)
If you want to retrain the model:
```bash
python train_model.py
```
This will generate `models/churn_model.pkl`.

### 5. Run the App
```bash
streamlit run app.py
```

## Usage
- Fill in the customer details in the sidebar.
- Click **Predict Churn** to see the prediction and probability.
- A bar chart will visualize the likelihood of churn.

## Model Details
- **Algorithm**: Random Forest Classifier
- **Features Used**: All features from the Telco Customer Churn dataset (except `customerID`)
- **Preprocessing**: Label encoding for categorical variables, missing value imputation for `TotalCharges`.

## File Descriptions
- `app.py`: Streamlit app for user interaction and prediction.
- `train_model.py`: Script to preprocess data, train the model, and save it as a pickle file.
- `data/telco_churn.csv`: Dataset for training.
- `models/churn_model.pkl`: Saved trained model.

## Requirements
- Python 3.7+
- streamlit
- pandas
- scikit-learn
- joblib
- matplotlib
- seaborn

## License
This project is for educational purposes.

## Acknowledgements
- [Telco Customer Churn Dataset](https://www.kaggle.com/blastchar/telco-customer-churn)
- Built with [Streamlit](https://streamlit.io/) and [scikit-learn](https://scikit-learn.org/)
