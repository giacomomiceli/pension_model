# Time series predictive modeling pipeline

## Instructions for usage:
1. Load your data into a pandas DataFrame
2. Ensure your data has the columns mentioned in your summary
3. Run: predictor, results, evaluation, comparison

```python
run_full_pipeline(your_dataframe)
```

4. Stress tests using given scenario

```python
# Instantiate the predictor
predictor = TimeSeriesPredictor(target_col='D_PIB')

# Make prediction on given scenario
predictions = predictor.predict_new_data(df_scenario)
```

### The pipeline will:
- [x] Perform comprehensive data analysis
- [x] Test multiple advanced ML models
- [x] Apply proper time series validation
- [x] Prevent overfitting through cross-validation
- [x] Provide detailed model diagnostics
- [x] Select the best performing model

---

## Appendix

### Advanced Model Selection
- Linear models (Ridge, Lasso, Elastic Net)
- Tree-based models (Random Forest, Gradient Boosting)
- Advanced gradient boosting (XGBoost, LightGBM)
- Support Vector Regression
- Neural Networks

### Proper Time Series Handling
- TimeSeriesSplit for cross-validation (maintains temporal order)
- Lag feature creation (dummy variables????) [to be improved]
- Rolling statistics
- Stationarity testing

### Overfitting Prevention
- Cross-validation with proper time series splits
- Hyperparameter tuning with `GridSearchCV`
- Feature selection using statistical tests
- Robust scaling for outlier handling

### Comprehensive Analysis
- Statistical tests (normality, stationarity)
- Correlation analysis
- Residual diagnostics
- Feature importance analysis

### Modern Libraries Used
- XGBoost & LightGBM: State-of-the-art gradient boosting
- Scikit-learn: Comprehensive ML toolkit
- Statsmodels: Advanced statistical analysis
- Seaborn/Matplotlib: Professional visualizations

### What You'll Get:
- Comprehensive data analysis with statistical tests
- Visual diagnostics including correlation heatmaps and residual plots
- Model comparison across different algorithms
- Best model selection based on cross-validation performance
- Feature importance ranking
- Prediction capabilities for new data