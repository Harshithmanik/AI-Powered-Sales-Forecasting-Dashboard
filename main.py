#!/usr/bin/env python3
"""
AI-Powered Sales Forecasting Dashboard - Main Execution Script

This script runs the complete sales forecasting analysis pipeline.
Author: Your Name
Date: 2024
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dashboard import SalesForecastingDashboard
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sales_forecasting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    """
    Main function to run the complete sales forecasting analysis
    """
    print("="*70)
    print("🚀 AI-POWERED SALES FORECASTING DASHBOARD")
    print("="*70)
    print("📊 Machine Learning-driven retail analytics solution")
    print("🎯 Generating insights and predictions for business growth")
    print("="*70)
    
    try:
        # Initialize the dashboard
        logging.info("Initializing Sales Forecasting Dashboard...")
        dashboard = SalesForecastingDashboard()
        
        # Run complete analysis
        logging.info("Starting complete analysis pipeline...")
        fig = dashboard.run_complete_analysis()
        
        # Save additional outputs
        logging.info("Saving analysis results...")
        
        # Save the forecast data
        if hasattr(dashboard, 'future_forecast'):
            dashboard.future_forecast.to_csv('outputs/future_forecast.csv', index=False)
            logging.info("✅ Future forecast saved to outputs/future_forecast.csv")
        
        # Save model performance metrics
        if hasattr(dashboard, 'feature_importance'):
            dashboard.feature_importance.to_csv('outputs/feature_importance.csv', index=False)
            logging.info("✅ Feature importance saved to outputs/feature_importance.csv")
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE!")
        print("="*70)
        print("📁 Generated Files:")
        print("   • sales_forecasting_dashboard.html - Interactive dashboard")
        print("   • outputs/future_forecast.csv - 90-day predictions")
        print("   • outputs/feature_importance.csv - Model insights")
        print("   • sales_forecasting.log - Execution log")
        print("\n🎯 Next Steps:")
        print("   1. Open sales_forecasting_dashboard.html in your browser")
        print("   2. Review business insights and recommendations")
        print("   3. Share results with stakeholders")
        print("="*70)
        
    except Exception as e:
        logging.error(f"Error during analysis: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
