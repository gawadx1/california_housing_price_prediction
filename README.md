# California Housing Price Prediction

This project builds a machine learning model to predict house prices in California using the **California Housing dataset**.

## 📂 Dataset
The dataset (`housing.csv`) contains information about:
- Longitude, Latitude
- Housing median age
- Total rooms, Total bedrooms
- Population, Households
- Median income
- Ocean proximity (categorical)
- Median house value (target variable)

## ⚙️ Workflow
1. Load and explore the dataset.
2. Preprocess:
   - Impute missing values.
   - Scale numerical features.
   - One-hot encode categorical features.
3. Train-test split (80/20).
4. Train a **RandomForestRegressor**.
5. Evaluate with:
   - RMSE (Root Mean Squared Error)
   - R² score
6. Analyze feature importances.
7. Save trained model.

## 📊 Results
- **RMSE**: ~49,000  
- **R²**: ~0.82  

The model explains about 82% of the variance in house prices.

## 🚀 How to Run
```bash
# Install dependencies
pip install pandas numpy scikit-learn matplotlib joblib

# Run the training script
python train_model.py
````

The trained model will be saved as:

```
/mnt/data/california_model.pkl
```

## 📈 Example Outputs

* Feature importance ranking
* Scatter plot of actual vs predicted house prices
* Sample predictions from the test set

---

### 🔮 Future Improvements

* Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
* Try XGBoost or LightGBM models
* Add engineered features (e.g., rooms per household)
