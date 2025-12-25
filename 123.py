import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score

np.random.seed(42)
n_samples = 500

age = np.random.randint(18, 60, n_samples)
stress = np.random.randint(1, 6, n_samples)
sleep = np.random.normal(7, 1.2, n_samples)
motivation = np.random.randint(1, 6, n_samples)

workout_hours = (motivation * 1.8) + (sleep * 0.7) - (stress * 0.4)
workout_hours += np.where(stress >= 4, -4.5, 0) 
workout_hours = np.clip(workout_hours + np.random.normal(0, 1.2, n_samples), 0, 20)

data = pd.DataFrame({
    'Age': age,
    'Stress': stress,
    'Sleep': sleep,
    'Motivation': motivation,
    'Workout': workout_hours
})

X = data[['Age', 'Stress', 'Sleep', 'Motivation']]
y = data['Workout']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)
svr = SVR()

model_list = [lr, rf, svr]
model_names = ['Linear Regression', 'Random Forest', 'SVR']
r2_results = []

for model in model_list:
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    r2_results.append(r2_score(y_test, p))

plt.figure(figsize=(9, 5))
sns.barplot(x=model_names, y=r2_results, palette='magma')
plt.ylabel('R2 Score')
plt.title('Model Comparison: Workout Prediction')
plt.ylim(0, 1)
for i, v in enumerate(r2_results):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center')
plt.show()

cluster_data = data[['Stress', 'Sleep', 'Workout']]
sc = StandardScaler()
z_data = sc.fit_transform(cluster_data)

km = KMeans(n_clusters=3, random_state=42)
data['Group'] = km.fit_predict(z_data)

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Stress', y='Workout', hue='Group', palette='Set1', s=80, alpha=0.8)
plt.title('Student Cluster Analysis')
plt.grid(True, alpha=0.3)
plt.show()

print("--- Cluster Summary ---")
print(data.groupby('Group')[['Stress', 'Sleep', 'Workout']].mean())