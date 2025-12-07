🩺 Diabetes Health Indicators – ML Prediction Web App

This project is an end-to-end Machine Learning + Flask web application that predicts the likelihood of diabetes using the Diabetes Health Indicators dataset.
It includes:

✔ ML pipeline (preprocessing + classifier)
✔ Imbalanced handling (SMOTE / class weights)
✔ Model saving with Joblib
✔ Flask Web UI for predictions
✔ Categorical → numeric mapping
✔ Form-based input and result display
🚀 Features

Clean end-to-end ML pipeline

Preprocessing + classifier wrapped in imblearn Pipeline

Categorical form inputs mapped via utils/mapper.py

Flask UI for prediction

Probability score & class label output

Fully portable project

🔧 Installation
1. Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

2. Create a virtual environment (recommended)
python -m venv env
source env/bin/activate    # Linux/Mac
env\Scripts\activate       # Windows

3. Install dependencies
pip install -r requirements.txt

🧠 Training the Model

## Run the training script:

python train_model.py

This script:

Loads + preprocesses the dataset

Handles class imbalance

Trains a model

# Saves the final pipeline to:

result/model.pkl
# 📘 API Details
POST /predict

## Form fields (21 ordered features):

HighBP, HighChol, CholCheck, BMI, Smoker, Stroke,
HeartDiseaseorAttack, PhysActivity, Fruits, Veggies,
HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth,
MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income


These are converted using:

# utils/mapper.py
Mapped list → pandas DataFrame → model.predict → result.
# 🗂 Key Files Explained
## utils/mapper.py

Handles mapping of categorical form elements:
Sex
GenHlth
Age categories
Education levels
Income groups
Ensures inputs match model training schema.
## app.py
Main Flask app:
Loads model.pkl
Renders index.html
Reads + maps form data
Predicts and returns result.html
# templates/index.html
Full user input form.
# templates/result.html
Displays:
Prediction
Probability score
# 📜 License
This project is free to use for learning and development.
