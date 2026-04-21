# import kagglehub
import pandas as pd
import os
from IPython.display import display
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Download dataset
# path = kagglehub.dataset_download("rohanrao/formula-1-world-championship-1950-2020")
path = r'C:\Users\ipanayo1\Desktop\Personal documents\МАГИСТЪР\Семестър 2\ИИСИ\formula-1-world-championship-1950-2020'
print("Path to dataset files:", path)

# --------------------------------------------------------------------------------------------

# Paths
results_csv_path = os.path.join(path, 'results.csv')
races_csv_path = os.path.join(path, 'races.csv')
drivers_csv_path = os.path.join(path, 'drivers.csv')
constructors_csv_path = os.path.join(path, 'constructors.csv')
qualifying_csv_path = os.path.join(path, 'qualifying.csv')

# Load data
results_df = pd.read_csv(results_csv_path)
races_df = pd.read_csv(races_csv_path)
drivers_df = pd.read_csv(drivers_csv_path)
constructors_df = pd.read_csv(constructors_csv_path)
qualifying_df = pd.read_csv(qualifying_csv_path)

# Merge datasets
merged_df = pd.merge(results_df, races_df, on='raceId')
merged_df = pd.merge(merged_df, drivers_df, on='driverId')
merged_df = pd.merge(merged_df, constructors_df, on='constructorId')
merged_df = pd.merge(merged_df, qualifying_df, on=['raceId', 'driverId'], how='left')

print('--- Merged DataFrame Head ---')
display(merged_df.head())

print('\n--- Merged DataFrame Info ---')
print(merged_df.info())


# --------------------------------------------------------------------------------------------

# Create driver name to ID mapping - dictionary
merged_df['driverName'] = merged_df['forename'] + ' ' + merged_df['surname']

driver_name_to_id = dict(zip(merged_df['driverName'], merged_df['driverId']))
# driver_id_to_name = dict(zip(merged_df['driverId'], merged_df['driverName']))

print('--- Driver Name to ID Mapping 5 Examples ---')
for name, driver_id in list(driver_name_to_id.items())[:5]:
    print(f"Driver Name: {name}, Driver ID: {driver_id}")

# --------------------------------------------------------------------------------------------

# Targets
merged_df['win'] = (merged_df['positionOrder'] == 1).astype(int)
merged_df['podium'] = (merged_df['positionOrder'] <= 3).astype(int)

y_position = merged_df['positionOrder']
y_time = merged_df['milliseconds']

print('--- Target Examples ---')
display(merged_df[['positionOrder', 'win', 'podium']].head())

# Features
features = [
    'grid',
    'laps',
    'points',
    'position_y',  # qualifying position
    'q1', 'q2', 'q3'
]

X = merged_df[features]
y_win = merged_df['win']
y_podium = merged_df['podium']

# Convert qualifying times to numeric (they are strings like "1:23.456")
def convert_time(val):
    try:
        if isinstance(val, str) and ':' in val:
            mins, secs = val.split(':')
            return float(mins) * 60 + float(secs)
        return float(val)
    except:
        return np.nan

for col in ['q1', 'q2', 'q3']:
    X[col] = X[col].apply(convert_time)

print('\n--- Missing Values Before ---')
print(X.isnull().sum())

# Fill missing values
X = X.fillna(X.mean())

print('\n--- Missing Values After ---')
print(X.isnull().sum())

print('\n--- Features Head ---')
display(X.head())

# --------------------------------------------------------------------------------------------

print('--- Training Model for Win Prediction ---')

X_train_win, X_test_win, y_train_win, y_test_win = train_test_split(
    X, y_win, test_size=0.2, random_state=42, stratify=y_win
)

model_win = RandomForestClassifier(n_estimators=100, random_state=42)
model_win.fit(X_train_win, y_train_win)

y_pred_win = model_win.predict(X_test_win)

print(f"Accuracy (Win): {accuracy_score(y_test_win, y_pred_win):.2f}")
print("Classification Report:\n", classification_report(y_test_win, y_pred_win))


print('\n--- Training Model for Podium Prediction ---')

X_train_pod, X_test_pod, y_train_pod, y_test_pod = train_test_split(
    X, y_podium, test_size=0.2, random_state=42, stratify=y_podium
)

model_podium = RandomForestClassifier(n_estimators=100, random_state=42)
model_podium.fit(X_train_pod, y_train_pod)

y_pred_pod = model_podium.predict(X_test_pod)

print(f"Accuracy (Podium): {accuracy_score(y_test_pod, y_pred_pod):.2f}")
print("Classification Report:\n", classification_report(y_test_pod, y_pred_pod))

# --------------------------------------------------------------------------------------------

print('--- Training Model for Position Prediction ---')
y_time = merged_df['milliseconds']

# Replace '\N' with NaN
y_time = y_time.replace('\\N', np.nan)

# Convert to numeric
y_time = pd.to_numeric(y_time)

# Fill missing values (или drop)
y_time = y_time.fillna(y_time.mean())
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(
    X, y_position, test_size=0.2, random_state=42
)

model_position = RandomForestRegressor(n_estimators=100, random_state=42)
model_position.fit(X_train_pos, y_train_pos)

y_pred_pos = model_position.predict(X_test_pos)

print(f"MAE (Position): {mean_absolute_error(y_test_pos, y_pred_pos):.2f}")
print(f"R2 (Position): {r2_score(y_test_pos, y_pred_pos):.2f}")


print('\n--- Training Model for Race Time Prediction ---')

X_train_time, X_test_time, y_train_time, y_test_time = train_test_split(
    X, y_time, test_size=0.2, random_state=42
)

model_time = RandomForestRegressor(n_estimators=100, random_state=42)
model_time.fit(X_train_time, y_train_time)

y_pred_time = model_time.predict(X_test_time)

print(f"MAE (Time): {mean_absolute_error(y_test_time, y_pred_time):.2f}")
print(f"R2 (Time): {r2_score(y_test_time, y_pred_time):.2f}")

# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------

def predict_head_to_head(driver1_name, driver2_name):

    # Check drivers exist
    if driver1_name not in driver_name_to_id or driver2_name not in driver_name_to_id:
        print("One or both drivers not found.")
        return

    driver1_id = driver_name_to_id[driver1_name]
    driver2_id = driver_name_to_id[driver2_name]

    # Get latest available data for both drivers
    d1 = merged_df[merged_df['driverId'] == driver1_id].iloc[-1:]
    d2 = merged_df[merged_df['driverId'] == driver2_id].iloc[-1:]

    if d1.empty or d2.empty:
        print("Not enough data for one of the drivers.")
        return

    # Extract features
    f1 = d1[features].copy()
    f2 = d2[features].copy()

    # Convert qualifying times
    for col in ['q1', 'q2', 'q3']:
        f1[col] = f1[col].apply(convert_time)
        f2[col] = f2[col].apply(convert_time)

    # Fill missing values
    f1 = f1.fillna(X.mean())
    f2 = f2.fillna(X.mean())

    # Predict positions
    pos1 = model_position.predict(f1)[0]
    pos2 = model_position.predict(f2)[0]

    # Predict win probability
    win1 = model_win.predict(f1)[0]
    win2 = model_win.predict(f2)[0]

    # Predict podium
    pod1 = model_podium.predict(f1)[0]
    pod2 = model_podium.predict(f2)[0]

    print(f"\n Head-to-Head: {driver1_name} vs {driver2_name}\n")

    print(f"{driver1_name}:")
    print(f"  Predicted Position: {round(pos1)}")
    print(f"  Win Chance: {'Yes' if win1 else 'No'}")
    print(f"  Top 3 Chance: {'Yes' if pod1 else 'No'}")

    print(f"\n{driver2_name}:")
    print(f"  Predicted Position: {round(pos2)}")
    print(f"  Win Chance: {'Yes' if win2 else 'No'}")
    print(f"  Top 3 Chance: {'Yes' if pod2 else 'No'}")

    # Decide winner
    print("\n Final Result:")

    if pos1 < pos2:
        print(f"Winner: {driver1_name}")
    elif pos2 < pos1:
        print(f"Winner: {driver2_name}")
    else:
        print("It's a tie (same predicted performance)")
        
# --------------------------------------------------------------------------------------------

driver1 = input("Enter driver 1 name: ")
driver2 = input("Enter driver 2 name: ")
predict_head_to_head(driver1,driver2)
# predict_head_to_head('Lewis Hamilton', 'Max Verstappen')
predict_head_to_head('Charles Leclerc', 'Lando Norris')
predict_head_to_head('Fernando Alonso', 'Sebastian Vettel')


# --------------------------------------------------------------------------------------------