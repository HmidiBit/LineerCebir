import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Dataset yükleme
df = pd.read_csv("dataset/creditcard.csv")

# Features ve target
X = df.drop("Class", axis=1)
y = df["Class"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA (Lineer Cebir kısmı)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X_pca,
    y,
    test_size=0.2,
    random_state=42
)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Tahmin
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Kaydet
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")

print("Model saved successfully.")