import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('youtube_ad_revenue.csv')
df = df.dropna()

features = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers', 'category', 'country']
target = 'ad_revenue_usd'

model_df = df[features + [target]].copy()
model_df.to_csv('cleaned_dataset.csv', index=False)
print(f"Cleaned dataset saved with {len(model_df)} records.")

X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers']
categorical_features = ['category', 'country']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features)
    ])

models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
}

results = []
trained_pipelines = {}

for name, model in models.items():
    print(f"Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', model)])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({'Model': name, 'R2 Score': r2, 'RMSE': rmse, 'MAE': mae})
    trained_pipelines[name] = pipeline

results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)
print(results_df)

best_model_name = results_df.iloc[0]['Model']
print(f"Best Model: {best_model_name}")
best_pipeline = trained_pipelines[best_model_name]

joblib.dump(best_pipeline, 'model.pkl')
print("Best model saved as 'model.pkl'.")
