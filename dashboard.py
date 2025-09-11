# AI-Powered Sales Forecasting Dashboard
# Complete solution with data generation, ML models, and interactive dashboard

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For machine learning and time series
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SalesForecastingDashboard:
    def __init__(self):
        self.data = None
        self.model = None
        self.predictions = None
        self.feature_importance = None
        
    def generate_sample_data(self):
        """Generate realistic retail sales data"""
        print("📊 Generating sample retail sales data...")
        
        # Date range: 3 years of historical data
        start_date = datetime(2021, 1, 1)
        end_date = datetime(2023, 12, 31)
        
        # Generate daily data
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Product categories and stores
        categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Toys']
        stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D', 'Store_E']
        regions = ['North', 'South', 'East', 'West', 'Central']
        
        # Generate data
        data = []
        np.random.seed(42)
        
        for date in date_range:
            for store in stores:
                for category in categories:
                    # Base sales with trends and seasonality
                    base_sales = np.random.normal(1000, 300)
                    
                    # Add seasonal patterns
                    month = date.month
                    day_of_week = date.weekday()
                    
                    # Holiday effects
                    if month == 12:  # December boost
                        base_sales *= 1.5
                    elif month in [6, 7]:  # Summer boost
                        base_sales *= 1.2
                    elif month in [1, 2]:  # Post-holiday dip
                        base_sales *= 0.8
                    
                    # Weekend effects
                    if day_of_week >= 5:  # Weekend
                        base_sales *= 1.1
                    
                    # Category-specific patterns
                    if category == 'Electronics' and month == 11:  # Black Friday
                        base_sales *= 2.0
                    elif category == 'Clothing' and month in [3, 4, 9, 10]:  # Season changes
                        base_sales *= 1.3
                    elif category == 'Toys' and month == 12:  # Christmas
                        base_sales *= 2.5
                    
                    # Ensure positive sales
                    sales = max(base_sales, 50)
                    
                    # Calculate other metrics
                    units_sold = int(sales / np.random.uniform(20, 100))
                    profit_margin = np.random.uniform(0.15, 0.35)
                    profit = sales * profit_margin
                    
                    data.append({
                        'Date': date,
                        'Store': store,
                        'Category': category,
                        'Region': regions[stores.index(store)],
                        'Sales': round(sales, 2),
                        'Units_Sold': units_sold,
                        'Profit': round(profit, 2),
                        'Profit_Margin': round(profit_margin, 2)
                    })
        
        self.data = pd.DataFrame(data)
        print(f"✅ Generated {len(self.data)} records")
        return self.data
    
    def feature_engineering(self):
        """Create features for machine learning"""
        print("🔧 Engineering features...")
        
        df = self.data.copy()
        
        # Date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Quarter'] = df['Date'].dt.quarter
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        
        # Lag features (previous sales)
        df = df.sort_values(['Store', 'Category', 'Date'])
        df['Sales_Lag_1'] = df.groupby(['Store', 'Category'])['Sales'].shift(1)
        df['Sales_Lag_7'] = df.groupby(['Store', 'Category'])['Sales'].shift(7)
        df['Sales_Lag_30'] = df.groupby(['Store', 'Category'])['Sales'].shift(30)
        
        # Rolling averages
        df['Sales_MA_7'] = df.groupby(['Store', 'Category'])['Sales'].rolling(7).mean().reset_index(0, drop=True)
        df['Sales_MA_30'] = df.groupby(['Store', 'Category'])['Sales'].rolling(30).mean().reset_index(0, drop=True)
        
        # Holiday indicators
        df['Is_Holiday_Season'] = ((df['Month'] == 12) | (df['Month'] == 11)).astype(int)
        df['Is_Summer'] = (df['Month'].isin([6, 7, 8])).astype(int)
        df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Encode categorical variables
        le_store = LabelEncoder()
        le_category = LabelEncoder()
        le_region = LabelEncoder()
        
        df['Store_Encoded'] = le_store.fit_transform(df['Store'])
        df['Category_Encoded'] = le_category.fit_transform(df['Category'])
        df['Region_Encoded'] = le_region.fit_transform(df['Region'])
        
        # Store encoders for later use
        self.encoders = {
            'store': le_store,
            'category': le_category,
            'region': le_region
        }
        
        self.data_features = df.dropna()  # Remove rows with NaN from lag features
        print(f"✅ Features engineered. Dataset shape: {self.data_features.shape}")
        
        return self.data_features
    
    def train_model(self):
        """Train machine learning model for forecasting"""
        print("🤖 Training machine learning model...")
        
        # Features for modeling
        feature_cols = [
            'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'WeekOfYear',
            'Store_Encoded', 'Category_Encoded', 'Region_Encoded',
            'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_30',
            'Sales_MA_7', 'Sales_MA_30',
            'Is_Holiday_Season', 'Is_Summer', 'Is_Weekend'
        ]
        
        X = self.data_features[feature_cols]
        y = self.data_features['Sales']
        
        # Split data (80% train, 20% test)
        split_date = self.data_features['Date'].quantile(0.8)
        train_mask = self.data_features['Date'] <= split_date
        
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        print(f"✅ Model trained!")
        print(f"📈 Train MAE: {train_mae:.2f}")
        print(f"📈 Test MAE: {test_mae:.2f}")
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Store predictions for visualization
        self.data_features.loc[train_mask, 'Predicted_Sales'] = train_pred
        self.data_features.loc[~train_mask, 'Predicted_Sales'] = test_pred
        
        return self.model
    
    def generate_future_forecast(self, days_ahead=90):
        """Generate future predictions"""
        print(f"🔮 Generating {days_ahead} days forecast...")
        
        # Get the last date in data
        last_date = self.data_features['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                   periods=days_ahead, freq='D')
        
        # Create future data structure
        future_data = []
        
        # Get unique combinations of Store, Category, Region
        unique_combinations = self.data_features[['Store', 'Category', 'Region', 
                                                'Store_Encoded', 'Category_Encoded', 
                                                'Region_Encoded']].drop_duplicates()
        
        for date in future_dates:
            for _, combo in unique_combinations.iterrows():
                # Basic date features
                future_row = {
                    'Date': date,
                    'Store': combo['Store'],
                    'Category': combo['Category'],
                    'Region': combo['Region'],
                    'Store_Encoded': combo['Store_Encoded'],
                    'Category_Encoded': combo['Category_Encoded'],
                    'Region_Encoded': combo['Region_Encoded'],
                    'Year': date.year,
                    'Month': date.month,
                    'Day': date.day,
                    'DayOfWeek': date.dayofweek,
                    'Quarter': (date.month - 1) // 3 + 1,
                    'WeekOfYear': date.isocalendar().week,
                    'Is_Holiday_Season': int(date.month in [11, 12]),
                    'Is_Summer': int(date.month in [6, 7, 8]),
                    'Is_Weekend': int(date.weekday() >= 5)
                }
                
                future_data.append(future_row)
        
        future_df = pd.DataFrame(future_data)
        
        # For simplicity, use average values for lag features and moving averages
        # In production, you'd implement a more sophisticated approach
        avg_sales = self.data_features.groupby(['Store', 'Category'])['Sales'].mean().reset_index()
        avg_sales_dict = {}
        for _, row in avg_sales.iterrows():
            key = (row['Store'], row['Category'])
            avg_sales_dict[key] = row['Sales']
        
        # Add lag features (simplified approach)
        for _, row in future_df.iterrows():
            key = (row['Store'], row['Category'])
            avg_val = avg_sales_dict.get(key, 1000)
            
            future_df.loc[_, 'Sales_Lag_1'] = avg_val * np.random.uniform(0.8, 1.2)
            future_df.loc[_, 'Sales_Lag_7'] = avg_val * np.random.uniform(0.8, 1.2)
            future_df.loc[_, 'Sales_Lag_30'] = avg_val * np.random.uniform(0.8, 1.2)
            future_df.loc[_, 'Sales_MA_7'] = avg_val * np.random.uniform(0.9, 1.1)
            future_df.loc[_, 'Sales_MA_30'] = avg_val * np.random.uniform(0.9, 1.1)
        
        # Make predictions
        feature_cols = [
            'Year', 'Month', 'Day', 'DayOfWeek', 'Quarter', 'WeekOfYear',
            'Store_Encoded', 'Category_Encoded', 'Region_Encoded',
            'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_30',
            'Sales_MA_7', 'Sales_MA_30',
            'Is_Holiday_Season', 'Is_Summer', 'Is_Weekend'
        ]
        
        future_predictions = self.model.predict(future_df[feature_cols])
        future_df['Predicted_Sales'] = future_predictions
        
        self.future_forecast = future_df
        print(f"✅ Future forecast generated!")
        
        return future_df
    
    def create_dashboard(self):
        """Create interactive dashboard with multiple visualizations"""
        print("📊 Creating interactive dashboard...")
        
        # Aggregate data by date for main trend line
        daily_sales = self.data_features.groupby('Date').agg({
            'Sales': 'sum',
            'Predicted_Sales': 'sum'
        }).reset_index()
        
        # Future forecast aggregated
        future_daily = self.future_forecast.groupby('Date')['Predicted_Sales'].sum().reset_index()
        future_daily.columns = ['Date', 'Future_Forecast']
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Sales Trend: Historical vs Predicted vs Future Forecast',
                'Sales by Category',
                'Sales by Store Performance',
                'Seasonal Pattern Analysis',
                'Feature Importance',
                'Monthly Sales Comparison'
            ],
            specs=[
                [{"colspan": 2}, None],
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Main trend line
        fig.add_trace(
            go.Scatter(
                x=daily_sales['Date'],
                y=daily_sales['Sales'],
                mode='lines',
                name='Actual Sales',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=daily_sales['Date'],
                y=daily_sales['Predicted_Sales'],
                mode='lines',
                name='Predicted Sales',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=future_daily['Date'],
                y=future_daily['Future_Forecast'],
                mode='lines',
                name='Future Forecast',
                line=dict(color='green', width=2, dash='dot')
            ),
            row=1, col=1
        )
        
        # 2. Sales by Category
        category_sales = self.data_features.groupby('Category')['Sales'].sum().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(
                x=category_sales.values,
                y=category_sales.index,
                orientation='h',
                name='Category Sales',
                marker_color='lightblue'
            ),
            row=2, col=1
        )
        
        # 3. Sales by Store
        store_sales = self.data_features.groupby('Store')['Sales'].sum().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(
                x=store_sales.values,
                y=store_sales.index,
                orientation='h',
                name='Store Sales',
                marker_color='lightcoral'
            ),
            row=2, col=2
        )
        
        # 4. Seasonal Pattern
        monthly_avg = self.data_features.groupby('Month')['Sales'].mean()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.add_trace(
            go.Bar(
                x=[month_names[i-1] for i in monthly_avg.index],
                y=monthly_avg.values,
                name='Monthly Average',
                marker_color='lightgreen'
            ),
            row=3, col=1
        )
        
        # 5. Feature Importance
        top_features = self.feature_importance.head(8)
        fig.add_trace(
            go.Bar(
                x=top_features['Importance'],
                y=top_features['Feature'],
                orientation='h',
                name='Feature Importance',
                marker_color='gold'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="🚀 AI-Powered Sales Forecasting Dashboard",
            title_font_size=24,
            showlegend=True,
            template='plotly_white'
        )
        
        # Update x-axis titles
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Sales ($)", row=2, col=1)
        fig.update_xaxes(title_text="Sales ($)", row=2, col=2)
        fig.update_xaxes(title_text="Month", row=3, col=1)
        fig.update_xaxes(title_text="Importance", row=3, col=2)
        
        # Update y-axis titles
        fig.update_yaxes(title_text="Sales ($)", row=1, col=1)
        fig.update_yaxes(title_text="Category", row=2, col=1)
        fig.update_yaxes(title_text="Store", row=2, col=2)
        fig.update_yaxes(title_text="Average Sales ($)", row=3, col=1)
        fig.update_yaxes(title_text="Feature", row=3, col=2)
        
        return fig
    
    def print_business_insights(self):
        """Generate business insights and recommendations"""
        print("\n" + "="*60)
        print("📈 BUSINESS INSIGHTS & RECOMMENDATIONS")
        print("="*60)
        
        # Calculate key metrics
        total_sales = self.data_features['Sales'].sum()
        avg_daily_sales = self.data_features.groupby('Date')['Sales'].sum().mean()
        
        # Best performing category
        best_category = self.data_features.groupby('Category')['Sales'].sum().idxmax()
        best_category_sales = self.data_features.groupby('Category')['Sales'].sum().max()
        
        # Best performing store
        best_store = self.data_features.groupby('Store')['Sales'].sum().idxmax()
        best_store_sales = self.data_features.groupby('Store')['Sales'].sum().max()
        
        # Seasonal insights
        peak_month = self.data_features.groupby('Month')['Sales'].mean().idxmax()
        peak_month_name = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][peak_month-1]
        
        # Future forecast insights
        future_total = self.future_forecast['Predicted_Sales'].sum()
        future_daily_avg = self.future_forecast.groupby('Date')['Predicted_Sales'].sum().mean()
        growth_rate = ((future_daily_avg - avg_daily_sales) / avg_daily_sales) * 100
        
        print(f"💰 Total Historical Sales: ${total_sales:,.2f}")
        print(f"📊 Average Daily Sales: ${avg_daily_sales:,.2f}")
        print(f"🏆 Best Category: {best_category} (${best_category_sales:,.2f})")
        print(f"🏪 Best Store: {best_store} (${best_store_sales:,.2f})")
        print(f"📅 Peak Season: {peak_month_name}")
        print(f"🔮 Future Daily Average: ${future_daily_avg:,.2f}")
        print(f"📈 Projected Growth: {growth_rate:+.1f}%")
        
        print(f"\n🎯 KEY RECOMMENDATIONS:")
        print(f"1. Focus marketing budget on {best_category} category")
        print(f"2. Replicate {best_store} success factors across other stores")
        print(f"3. Prepare inventory buildup for {peak_month_name} peak season")
        print(f"4. Expected {'growth' if growth_rate > 0 else 'decline'} of {abs(growth_rate):.1f}% in coming months")
        
        # Top features
        print(f"\n🔍 TOP PREDICTIVE FACTORS:")
        for i, row in self.feature_importance.head(5).iterrows():
            print(f"{i+1}. {row['Feature']}: {row['Importance']:.3f}")
        
        print("="*60)
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting AI-Powered Sales Forecasting Analysis...")
        print("="*60)
        
        # Step 1: Generate data
        self.generate_sample_data()
        
        # Step 2: Feature engineering
        self.feature_engineering()
        
        # Step 3: Train model
        self.train_model()
        
        # Step 4: Generate forecast
        self.generate_future_forecast()
        
        # Step 5: Create dashboard
        dashboard_fig = self.create_dashboard()
        
        # Step 6: Print insights
        self.print_business_insights()
        
        print("\n✅ Analysis Complete!")
        print("📊 Dashboard created successfully!")
        
        # Save the dashboard as HTML
        dashboard_fig.write_html("sales_forecasting_dashboard.html")
        print("💾 Dashboard saved as 'sales_forecasting_dashboard.html'")
        
        # Show the dashboard
        dashboard_fig.show()
        
        return dashboard_fig

# Run the complete analysis
if __name__ == "__main__":
    dashboard = SalesForecastingDashboard()
    fig = dashboard.run_complete_analysis()
    
    # Create a summary statistics table
    print("\n" + "="*50)
    print("📋 DATASET SUMMARY")
    print("="*50)
    print(dashboard.data.describe())
    
    print("\n📊 DATA SAMPLE:")
    print(dashboard.data.head())
    
    print(f"\n📈 Model Performance Metrics:")
    print(f"Features used: {len(dashboard.feature_importance)} features")
    print(f"Training data points: {len(dashboard.data_features)}")
    print(f"Forecast period: 90 days")
    print(f"Categories analyzed: {dashboard.data['Category'].nunique()}")
    print(f"Stores analyzed: {dashboard.data['Store'].nunique()}")
