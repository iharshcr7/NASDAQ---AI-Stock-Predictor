import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("data/stock_market_dataset/stocks/AAPL.csv")

# Create target
df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# Drop last row
df = df.dropna()

# Balance dataset
up = df[df['target'] == 1]
down = df[df['target'] == 0].sample(len(up), random_state=42)

df = pd.concat([up, down]).sample(frac=1, random_state=42)

# Features
X = df[['Open','High','Low','Volume']]
y = df['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained successfully")