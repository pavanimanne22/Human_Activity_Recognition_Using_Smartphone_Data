
  # 🤖 Human Activity Recognition Using Smartphone Sensor Data

This project leverages smartphone sensor data (accelerometers and gyroscopes) to recognize and classify human activities using machine learning. The goal is to identify actions such as walking, sitting, standing, and lying down using time-series data from smartphones.

---

## 📚 Project Overview

Human Activity Recognition (HAR) is an important field in health monitoring, sports, elderly care, and smart devices. This project uses a public HAR dataset and applies machine learning techniques to build a classifier capable of predicting activities with high accuracy.

---

## ✅ Key Features

- 📊 Exploratory Data Analysis (EDA) on time-series features
- 🧹 Data cleaning and preprocessing
- 🔍 Feature selection and label encoding
- ⚙️ Built and evaluated classification models:
  - **Random Forest**
  - **Support Vector Machine (SVM)**
  - **K-Nearest Neighbors (KNN)**
- 📈 Performance metrics: Accuracy, Precision, Recall, F1-score
- 📉 Confusion matrix visualization

---

## 🛠️ Tech Stack

| Component       | Tools/Libraries                             |
|------------------|---------------------------------------------|
| Language         | Python                                      |
| Data Handling    | Pandas, NumPy                               |
| Visualization    | Matplotlib, Seaborn                         |
| Machine Learning | Scikit-learn                                |
| Environment      | Jupyter Notebook                            |

---

## 📁 Folder Structure

human-activity-recognition/
├── human_activity_recognition_using_smartphone_Y.ipynb # Jupyter notebook
├── dataset/ # Dataset files (not uploaded here)
├── README.md # Project documentation
├── requirements.txt # Python dependencies




##Install dependencies

pip install -r requirements.txt



✅ 1. Install All Packages Manually (One Line)
You can install all libraries by running:

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
✅ 2. Use a Virtual Environment (Recommended for Clean Projects)
Keeps dependencies isolated from your global environment.

🔹 Steps:

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3.Install packages
pip install numpy pandas matplotlib seaborn scikit-learn jupyter


##✅ 4. Use Conda (if using Anaconda)
If you are working in Anaconda:
conda create -n har_env python=3.10
conda activate har_env
conda install numpy pandas matplotlib seaborn scikit-learn
conda install -c conda-forge notebook


#✅ 5. Install Within Jupyter Notebook (Quick Testing)
##Use %pip directly inside a notebook cell:
%pip install numpy pandas matplotlib seaborn scikit-learn


##💡 Tip:
To save all installed packages in a file later:
pip freeze > requirements.txt



##Launch the Jupyter Notebook
jupyter notebook
Open human_activity_recognition_using_smartphone_Y.ipynb and run all cells.

## 📈 Model Results

| Model          | Accuracy | F1 Score |
|----------------|----------|----------|
| Random Forest  | ~95%     | ~94%     |
| SVM            | ~93%     | ~92%     |
| KNN            | ~91%     | ~90%     |

