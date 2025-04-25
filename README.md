# 💰 IntelliWealth – AI-Powered Investment Intelligence

IntelliWealth is a next-generation online platform that empowers **individual investors** and **financial advisors** to make intelligent, data-driven investment decisions. It leverages **artificial intelligence**, **natural language processing**, and **interactive visualizations** to create a personalized investment journey tailored to each user's goals, risk tolerance, and real-time market conditions.


## 🚀 Key Features

### 📊 Dynamic Asset Allocation
- Allocates assets dynamically based on user-defined goals and real-time financial indicators.
- Incorporates historical volatility and correlation analysis.
- Utilizes **LSTM** models for accurate stock price predictions.

### 🔮 Predictive Market Analytics
- Conducts sentiment analysis on financial tweets using advanced models like **FinBERT** and **LDA** to gauge market sentiment.

### 📈 Interactive Visualizations
- Employs **Chart.js** for creating rich, customizable visual insights into portfolio performance and market stress-testing.

### 🖥️ Centralized Dashboard
- Features a unified dashboard with **real-time notifications**, comprehensive performance tracking, and personalized investment recommendations.

---

## 🛠️ System Architecture

Modular and scalable system composed of:

- **AI Modeling Engine:** asset allocation and Forecasting.
- **NLP Pipeline:** Sentiment extraction using FinBERT and Latent Dirichlet Allocation (LDA).
- **Visualization Module:** Chart.js for real-time charts and simulations.
- **Backend:** Django
- **Database:** PostgreSQL for secure, relational data management.
- **Frontend:** [React.js](https://reactjs.org/) 
---

## ⚙️ Installation & Setup

### Prerequisites

- Python 3.9+
- Node.js 14+
- PostgreSQL

# Project Setup Instructions

## Backend Setup

### Create virtual environment and activate it
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install all the requirements
```bash
pip install -r requirements.txt
```

### Start the Django backend server 
Backend will start on http://127.0.0.1:8000/
```bash
python manage.py makemigrations
python manage.py migrate
python manage.py createsuperuser  # Follow prompts to create admin account
python manage.py runserver
```

## Frontend Setup
Following will start the frontend on http://localhost:3000/landing
```bash
cd frontend
npm i
npm start
```

