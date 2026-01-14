import kagglehub
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from kagglehub import KaggleDatasetAdapter
import pandas as pd

def load_data(dataset_path: str, file_path: str) -> pd.DataFrame:
    return kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        dataset_path,
        file_path
    )

def get_strongest_correlations(df: pd.DataFrame, target_column: str, n: int = 5) -> pd.Series:
    correlations = df.corr(numeric_only = True)[target_column].abs().sort_values(ascending=False)
    return correlations[1:n+1]


def main():

    training_df_1 = load_data("parthdande/nba-mvp-voting-dataset-2000-2021", "2001-2010 MVP Data.csv").dropna()
    training_df_2 = load_data("parthdande/nba-mvp-voting-dataset-2000-2021", "2010-2021 MVP Data.csv").dropna()
    test_data = load_data("parthdande/nba-mvp-voting-dataset-2000-2021", "2022-2023 MVP Data.csv").dropna()
    team_data = load_data("sumitrodatta/nba-aba-baa-stats", "Team Summaries.csv").dropna()

    team_data = team_data.rename(columns={
        "abbreviation": "Tm",
        "season": "year"
    })
    training_df = pd.concat([training_df_1, training_df_2], ignore_index=True)
    training_df = pd.merge(training_df, team_data, on=["Tm", "year"], how='inner')
    training_df = training_df.drop(columns=["Share", "Unnamed: 0",
                                           "Pts Max", "First", "year"])
    test_data = test_data[test_data["year"] == 2023]
    test_data = pd.merge(test_data, team_data, on=["Tm", "year"], how='inner')

    features = get_strongest_correlations(training_df, target_column="Pts Won", n=10)
    print("### Selected Features based on Correlation with 'Pts Won' ###")
    print(features)

    X_train = training_df[features.index.tolist()]
    y_train = training_df["Pts Won"]
    
    X_test = test_data[features.index.tolist()]
    y_test = test_data["Pts Won"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_estimator = DecisionTreeRegressor(max_depth=1, splitter='best', min_samples_split=2)
    model = AdaBoostRegressor(estimator=base_estimator, n_estimators=10)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    # 1. Create a DataFrame to hold the results
    # We use test_data for the Player and Team info
    results_df = test_data.copy() 
    
    # 2. Add the true and predicted values
    results_df['True_Pts_Won'] = y_test.values
    results_df['Predicted_Pts_Won'] = y_pred

    # 3. Calculate the prediction error (Residual)
    results_df['Error'] = results_df['True_Pts_Won'] - results_df['Predicted_Pts_Won']

    # 4. Sort the results by the highest actual points won to see the top contenders first
    # Or, sort by predicted points to see who the model thinks should have won
    results_df = results_df.sort_values(by='Predicted_Pts_Won', ascending=False)
    
    # 5. Select and print the relevant columns (e.g., top 10 players)
    print("### Model Predictions vs. True Values (Top 10 Predictions) ###")
    print(results_df[['Player', 'Tm', 'True_Pts_Won', 'Predicted_Pts_Won', 'Error']].head(10).to_markdown(index=False))

    # Calculate and print the overall performance metric
    mse = mean_squared_error(y_test, y_pred)
    print(f"\nMean Squared Error (MSE): {mse:.4f}")




if __name__ == "__main__":
    main()
