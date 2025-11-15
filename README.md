# 🚀 AI-Powered Sales Forecasting Dashboard

A comprehensive machine learning solution for retail sales forecasting with interactive dashboard visualization.

<img width="1862" height="1219" alt="Screenshot 2025-09-11 100104" src="https://github.com/user-attachments/assets/5e85fdbd-b960-474b-90f0-183c77ec120a" />

<img width="1864" height="1051" alt="Screenshot 2025-09-11 100129" src="https://github.com/user-attachments/assets/d55fddb3-f668-42e2-a212-6657e01fc674" />

<img width="1861" height="828" alt="Screenshot 2025-09-11 100210" src="https://github.com/user-attachments/assets/6a214262-c07f-436a-b986-c97c2421608a" />



## 🎯 Project Overview

This project demonstrates a complete end-to-end data science workflow for sales forecasting, combining:
- **Machine Learning**: Random Forest algorithm for time series prediction
- **Feature Engineering**: 17+ engineered features including seasonality, trends, and lag variables
- **Interactive Dashboard**: Real-time visualizations built with React and Recharts
- **Business Intelligence**: Automated insights and actionable recommendations

## 🛠️ Technologies Used

### Backend
- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **Plotly** - Interactive visualizations
- **NumPy** - Numerical computations

### Frontend
- **React** - Interactive dashboard framework
- **Recharts** - Chart library for data visualization
- **Tailwind CSS** - Styling and responsive design

## 📊 Key Features

✅ **Time Series Forecasting** - 90-day sales predictions  
✅ **Multi-dimensional Analysis** - By category, store, and region  
✅ **Seasonal Pattern Detection** - Holiday and weekend effects  
✅ **Performance Metrics** - MAE, RMSE, and accuracy scores  
✅ **Interactive Visualizations** - Hover tooltips and dynamic filtering  
✅ **Business Insights** - Automated recommendations and KPIs  
✅ **Real-time Dashboard** - Live data updates and forecasting  

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Node.js 14+
npm or yarn
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-sales-forecasting-dashboard.git
cd ai-sales-forecasting-dashboard
```

2. **Set up Python environment**
```bash
pip install -r requirements.txt
```

3. **Run the main analysis**
```bash
python src/main.py
```

4. **Set up React dashboard** (optional)
```bash
cd frontend
npm install
npm start
```

## 📈 Usage

### Running the Full Analysis
```python
from src.dashboard import SalesForecastingDashboard

# Initialize and run complete analysis
dashboard = SalesForecastingDashboard()
fig = dashboard.run_complete_analysis()
```

### Custom Data Input
```python
# Load your own data
dashboard.data = pd.read_csv('your_sales_data.csv')
dashboard.feature_engineering()
dashboard.train_model()
dashboard.generate_future_forecast()
```

## 📊 Sample Output

The dashboard generates:
- **Sales trend analysis** with historical vs predicted data
- **Category performance** rankings and insights
- **Store comparison** and best practices identification  
- **Seasonal forecasting** with holiday effect modeling
- **Business recommendations** based on ML insights

## 🎯 Business Impact

### Key Metrics Tracked:
- **Total Sales Volume**: $500K+ analyzed
- **Forecast Accuracy**: 95%+ prediction accuracy
- **Growth Analysis**: Monthly and yearly trend identification
- **Peak Season Detection**: Holiday and seasonal pattern analysis

### Actionable Insights:
1. **Category Optimization**: Focus on high-performing product lines
2. **Store Performance**: Replicate best practices across locations
3. **Inventory Planning**: Prepare for seasonal demand spikes
4. **Marketing Strategy**: Target campaigns during peak periods

## 📁 Project Structure

```
├── src/
│   ├── main.py              # Main execution script
│   ├── dashboard.py         # Core dashboard class
│   └── utils.py            # Utility functions
├── notebooks/
│   └── analysis.ipynb      # Jupyter notebook for exploration
├── frontend/
│   └── src/
│       └── Dashboard.js    # React dashboard component
├── data/
│   └── sample_data/        # Sample datasets
└── screenshots/
    └── dashboard_preview.png
```

## 🔍 Model Details

### Algorithm: Random Forest Regressor
- **Features**: 17 engineered variables
- **Training Data**: 3 years historical sales
- **Validation**: Time series cross-validation
- **Performance**: MAE < 5%, RMSE optimized

### Feature Engineering:
- **Temporal Features**: Year, month, day, week patterns
- **Lag Features**: 1-day, 7-day, 30-day historical values
- **Moving Averages**: 7-day and 30-day rolling windows
- **Seasonal Indicators**: Holiday, weekend, summer flags
- **Category Encoding**: Store, product, and region encoding


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request




## 📞 Contact

K.HARSHITH - harshithmanik2004@gmail.com  
**Project Link**: https://github.com/Harshithmanik/AI-Powered-sales-Forecasting-Dashboard

---
⭐ **Star this repo if you found it helpful!** ⭐
