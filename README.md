# Web CSV Analyzer

**Web CSV Analyzer** is a Streamlit-powered web application designed to help you upload, explore, clean, and analyze CSV datasets seamlessly. With built-in data profiling, visualizations, statistical tests, and export options, it's your go-to tool for quick and effective data analysis.

---

## 🚀 Features

-  **Upload CSV Files** – Drag and drop or select local CSV files for analysis
-  **Data Cleaning** – Handle missing values, remove duplicates, and convert data types
-  **Exploratory Data Analysis (EDA)** – Generate summary statistics, distribution plots, and box plots
-  **Advanced Visualizations** – Correlation heatmaps, interactive scatter plots, time series, and more
-  **Statistical Analysis** – Correlation tests, linear regression, and basic ML models
-  **Export Cleaned Data** – Download final dataset and auto-generated reports

---

## 🛠️ Tech Stack

- **Frontend/UI**: `Streamlit`
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `seaborn`, `matplotlib`, `plotly`
- **Modeling**: `scikit-learn`, `scipy`
- **API Integration** *(optional)*: Gemini API for report insights 

---

## 📦 Setup Instructions

1. **Clone the Repo**
```bash
git clone https://github.com/chaanakyaaM/Web-CSV-Analyzer.git
cd web-csv-analyzer
```

2. **Create Virtual Environment** (optional but recommended)
```bash
python -m venv venv
venv/Scripts/activate  # or source venv\bin\activate on Linux
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up .env file** (optional for Gemini integration)
```env
MODEL_NAME=gemini-pro
```

5. **Run the App**
```bash
streamlit run app.py
```


---

## 🧠 Use Cases
- Performing simple data analysis
- Data preprocessing for ML projects
- Quick EDA on messy datasets
- Teaching data cleaning concepts visually
- Generating visual/statistical reports from raw files
- Insights report generation using AI

---

## 🤖 Optional: Gemini-Powered Reports
You can plug in the Gemini API key to generate summaries of your dataset's quality, completeness, and key insights — perfect for automated reporting!

---

## ✨ Credits
Created with ❤️ by [chaanakyaa M](https://github.com/chaanakyaaM)

---

