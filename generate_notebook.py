import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# YouTube Monetization Modeler\n",
    "This notebook covers the end-to-end process of training a machine learning model to predict YouTube ad revenue based on video performance metrics."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Import Required Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
    "from sklearn.compose import ColumnTransformer\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n",
    "from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor\n",
    "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error\n",
    "import joblib\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Load the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_path = 'youtube_ad_revenue.csv'\n",
    "df = pd.read_csv(data_path)\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Exploratory Data Analysis & Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(df.info())\n",
    "print(df.describe())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop rows with missing values if any\n",
    "df = df.dropna()\n",
    "\n",
    "# Select the features we want to use for training\n",
    "# target: 'ad_revenue_usd'\n",
    "features = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers', 'category', 'country']\n",
    "target = 'ad_revenue_usd'\n",
    "\n",
    "# Create final dataframe\n",
    "model_df = df[features + [target]].copy()\n",
    "model_df.to_csv('cleaned_dataset.csv', index=False)\n",
    "print(f\"Cleaned dataset saved with {len(model_df)} records.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Preprocessing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = model_df[features]\n",
    "y = model_df[target]\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "numeric_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'video_length_minutes', 'subscribers']\n",
    "categorical_features = ['category', 'country']\n",
    "\n",
    "preprocessor = ColumnTransformer(\n",
    "    transformers=[\n",
    "        ('num', StandardScaler(), numeric_features),\n",
    "        ('cat', 'passthrough', categorical_features) # using codes as provided\n",
    "    ])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Model Training and Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "models = {\n",
    "    'Linear Regression': LinearRegression(),\n",
    "    'Ridge Regression': Ridge(),\n",
    "    'Lasso Regression': Lasso(),\n",
    "    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1),\n",
    "    'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)\n",
    "}\n",
    "\n",
    "results = []\n",
    "trained_pipelines = {}\n",
    "\n",
    "for name, model in models.items():\n",
    "    print(f\"Training {name}...\")\n",
    "    pipeline = Pipeline(steps=[('preprocessor', preprocessor),\n",
    "                               ('regressor', model)])\n",
    "    \n",
    "    pipeline.fit(X_train, y_train)\n",
    "    y_pred = pipeline.predict(X_test)\n",
    "    \n",
    "    r2 = r2_score(y_test, y_pred)\n",
    "    rmse = np.sqrt(mean_squared_error(y_test, y_pred))\n",
    "    mae = mean_absolute_error(y_test, y_pred)\n",
    "    \n",
    "    results.append({'Model': name, 'R2 Score': r2, 'RMSE': rmse, 'MAE': mae})\n",
    "    trained_pipelines[name] = pipeline\n",
    "\n",
    "results_df = pd.DataFrame(results).sort_values(by='R2 Score', ascending=False)\n",
    "results_df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Model Interpretation and Export"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "best_model_name = results_df.iloc[0]['Model']\n",
    "print(f\"Best Model: {best_model_name}\")\n",
    "best_pipeline = trained_pipelines[best_model_name]\n",
    "\n",
    "joblib.dump(best_pipeline, 'model.pkl')\n",
    "print(\"Best model saved as 'model.pkl'.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Feature Importance (if applicable)\n",
    "if hasattr(best_pipeline.named_steps['regressor'], 'feature_importances_'):\n",
    "    importances = best_pipeline.named_steps['regressor'].feature_importances_\n",
    "    \n",
    "    # Note: If OneHotEncoder is used, feature names are dynamic. Since we use passthrough, they remain the same.\n",
    "    feature_names = numeric_features + categorical_features\n",
    "    \n",
    "    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})\n",
    "    imp_df = imp_df.sort_values(by='Importance', ascending=False)\n",
    "    \n",
    "    plt.figure(figsize=(10, 6))\n",
    "    sns.barplot(x='Importance', y='Feature', data=imp_df, palette='viridis')\n",
    "    plt.title('Feature Importances - ' + best_model_name)\n",
    "    plt.show()\n",
    "else:\n",
    "    print(\"Feature importance not supported by the best model.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Actionable Insights for Content Creators\n",
    "Based on the feature importances extracted above, content creators should focus heavily on the most impactful features. Usually, `watch_time_minutes` and `views` dominate ad revenue. Increasing engagement via likes and comments also contributes to the video's profitability."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open("model_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
print("Notebook generated successfully.")
