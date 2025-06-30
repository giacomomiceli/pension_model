# Time series predictive modeling pipeline

## Instructions for usage:
1. Load your data into a pandas DataFrame
2. Ensure your data has the columns mentioned in your summary
3. Run: predictor, results, evaluation, comparison

```python
predictor, results, evaluation, comparison = run_full_pipeline(your_dataframe)
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

## Model training and presitions

### Advanced model selection
- Linear models (Ridge, Lasso, Elastic Net)
- Tree-based models (Random Forest, Gradient Boosting)
- Advanced gradient boosting (XGBoost, LightGBM)
- Support Vector Regression
- Neural Networks

### Proper time series handling
- TimeSeriesSplit for cross-validation (maintains temporal order)
- Lag feature creation (dummy variables????) [to be improved]
- Rolling statistics
- Stationarity testing

### Overfitting prevention
- Cross-validation with proper time series splits
- Hyperparameter tuning with `GridSearchCV`
- Feature selection using statistical tests
- Robust scaling for outlier handling

### Comprehensive analysis
- Statistical tests (normality, stationarity)
- Correlation analysis
- Residual diagnostics
- Feature importance analysis

### Modern libraries used
- `XGBoost` and `LightGBM`: State-of-the-art gradient boosting
- `Scikit-learn`: Comprehensive ML toolkit
- `Statsmodels`: Advanced statistical analysis
- `Seaborn/Matplotlib` and `Plotly`: Professional visualizations

### What You'll Get:
- Comprehensive data analysis with statistical tests
- Visual diagnostics including correlation heatmaps and residual plots
- Model comparison across different algorithms
- Best model selection based on cross-validation performance
- Feature importance ranking
- Prediction capabilities for new data

---

# Improvements

## Dummy variables
As Economic relationships often change during crises or policy shifts, for economic data, one shoould consider dummy variables:

- Economic crisis periods: 
    - 2007-2009 (Financial Crisis), 
    - 2010-2012 (European Debt Crisis), 
    - 2020-2022 (COVID-19)
- Policy regime changes: Different economic policy periods
- Structural breaks: High/low volatility periods in economic indicators
- (Seasonal effects: If you have quarterly/monthly data)

## Feature significance analysis
Check/develop comprehensive_feature_analysis() (check in sandbox) that provides:

- Correlation analysis: Linear relationships
- Mutual information: Non-linear relationships
- F-statistics: Statistical significance testing
- Univariate R-squared: Individual predictive power
- Composite ranking: Combined score across all metrics
- Visual comparisons: Charts showing feature importance

This would give a better data-driven ranking of which features are most important for predicting `D_PIB`.

```python
# Run the enhanced pipeline
predictor, results, evaluation, comparison, feature_analysis, feature_summary, dummy_recommendations = run_full_pipeline(df)

# Get the most important features
print("Top 5 most predictive features:")
print(feature_summary.head(5))

# Review dummy variable suggestions
print("Dummy variable recommendations:")
for rec in dummy_recommendations:
    print(f"- {rec['variable']}: {rec['recommendation']}")
```