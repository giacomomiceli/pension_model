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

## The pipeline will:
- [x] Perform comprehensive data analysis
- [x] Test multiple advanced ML models
- [x] Apply proper time series validation
- [x] Prevent overfitting through cross-validation
- [x] Provide detailed model diagnostics
- [x] Select the best performing model