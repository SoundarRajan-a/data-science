# ===== CUSTOMER PURCHASE FREQUENCY ANALYSIS =====

# 1. DATA LOADING AND EXPLORATION
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import gradio as gr
import warnings
warnings.filterwarnings('ignore')

# Load the dataset (replace with actual data source)
# df = pd.read_csv('customer_data.csv')

# For demonstration, create a sample dataset
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'customer_id': range(1, n_samples + 1),
    'age': np.random.randint(18, 80, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'total_purchases': np.random.randint(1, 100, n_samples),
    'total_spend': np.random.randint(100, 10000, n_samples),
    'avg_order_value': np.random.randint(20, 500, n_samples),
    'customer_tenure_months': np.random.randint(1, 120, n_samples),
    'website_visits': np.random.randint(0, 500, n_samples),
    'cart_abandonment_rate': np.random.uniform(0, 100, n_samples),
    'product_preference': np.random.choice(['Electronics', 'Fashion', 'Home', 'Beauty'], n_samples),
    'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_samples),
    'newsletter_subscribed': np.random.choice(['Yes', 'No'], n_samples),
    'customer_reviews': np.random.randint(0, 50, n_samples),
    'last_purchase_days_ago': np.random.randint(0, 365, n_samples)
})

# Create target variable based on total purchases
def categorize_frequency(purchases):
    if purchases < 5:
        return 'Rare'
    elif purchases < 15:
        return 'Occasional'
    elif purchases < 40:
        return 'Regular'
    else:
        return 'Frequent'

df['purchase_frequency'] = df['total_purchases'].apply(categorize_frequency)

# Display basic info
print("Dataset Shape:", df.shape)
print("\nFirst Few Rows:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check for duplicates
print("\nDuplicate Rows:", df.duplicated().sum())

# ===== EXPLORATORY DATA ANALYSIS =====

# 1. Distribution of Purchase Frequency
plt.figure(figsize=(10, 5))
df['purchase_frequency'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Distribution of Customer Purchase Frequency')
plt.xlabel('Frequency Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# 2. Age Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['age'], kde=True, color='green')
plt.title('Distribution of Customer Age')
plt.xlabel('Age')
plt.show()

# 3. Total Spend Distribution
plt.figure(figsize=(10, 5))
sns.histplot(df['total_spend'], kde=True, color='orange')
plt.title('Distribution of Total Customer Spend')
plt.xlabel('Total Spend ($)')
plt.show()

# 4. Relationship between Total Spend and Purchase Frequency
plt.figure(figsize=(12, 6))
sns.boxplot(x='purchase_frequency', y='total_spend', data=df)
plt.title('Total Spend by Purchase Frequency Category')
plt.xlabel('Purchase Frequency')
plt.ylabel('Total Spend ($)')
plt.show()

# 5. Correlation Heatmap
numeric_df = df[['age', 'total_purchases', 'total_spend', 'avg_order_value',
                   'customer_tenure_months', 'website_visits', 'cart_abandonment_rate']]
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()

# 6. Customer Tenure vs Purchase Frequency
plt.figure(figsize=(12, 6))
sns.boxplot(x='purchase_frequency', y='customer_tenure_months', data=df)
plt.title('Customer Tenure by Purchase Frequency Category')
plt.xlabel('Purchase Frequency')
plt.ylabel('Tenure (Months)')
plt.show()

# ===== FEATURE ENGINEERING =====

# Create new features
df['customer_lifetime_value'] = df['total_spend'] / (df['customer_tenure_months'] + 1)
df['purchase_velocity'] = df['total_purchases'] / (df['customer_tenure_months'] + 1)
df['recent_purchase'] = (df['last_purchase_days_ago'] <= 30).astype(int)
df['review_engagement_rate'] = df['customer_reviews'] / (df['total_purchases'] + 1)
df['engagement_score'] = (df['website_visits'] / 100) + (df['customer_reviews'] / 10)

print("\nNew Features Created:")
print(df[['customer_lifetime_value', 'purchase_velocity', 'recent_purchase']].head())

# ===== DATA PREPROCESSING =====

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'product_preference', 'device_type', 'newsletter_subscribed']

df_encoded = df.copy()

for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Drop non-numeric and target columns
X = df_encoded.drop(['customer_id', 'purchase_frequency'], axis=1)
y = df_encoded['purchase_frequency']

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print("\nFeature Columns:", X.columns.tolist())
print("Target Classes:", le_target.classes_)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded,
                                                      test_size=0.2, random_state=42)

print(f"\nTraining Set Size: {X_train.shape}")
print(f"Test Set Size: {X_test.shape}")

# ===== MODEL BUILDING AND EVALUATION =====

# 1. Logistic Regression (Baseline)
print("\n" + "="*50)
print("LOGISTIC REGRESSION MODEL")
print("="*50)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred, average='weighted', zero_division=0)
lr_recall = recall_score(y_test, lr_pred, average='weighted', zero_division=0)
lr_f1 = f1_score(y_test, lr_pred, average='weighted', zero_division=0)

print(f"Accuracy: {lr_accuracy:.4f}")
print(f"Precision: {lr_precision:.4f}")
print(f"Recall: {lr_recall:.4f}")
print(f"F1-Score: {lr_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

print("\nClassification Report:")
print(classification_report(y_test, lr_pred, target_names=le_target.classes_))

# 2. Random Forest Classifier (Advanced)
print("\n" + "="*50)
print("RANDOM FOREST CLASSIFIER MODEL")
print("="*50)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_recall = recall_score(y_test, rf_pred, average='weighted', zero_division=0)
rf_f1 = f1_score(y_test, rf_pred, average='weighted', zero_division=0)

print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1-Score: {rf_f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred, target_names=le_target.classes_))

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10), palette='viridis')
plt.title('Top 10 Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.show()

# ===== GRADIO DEPLOYMENT =====

# Best model selection (Random Forest)
best_model = rf_model
best_scaler = scaler
best_le_target = le_target

def predict_purchase_frequency(age, gender, total_purchases, total_spend,
                               avg_order_value, tenure_months, website_visits,
                               cart_abandonment_rate, product_preference,
                               device_type, newsletter, reviews, last_purchase_days):
    """
    Predict customer purchase frequency based on input features
    """

    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'total_purchases': [total_purchases],
        'total_spend': [total_spend],
        'avg_order_value': [avg_order_value],
        'customer_tenure_months': [tenure_months],
        'website_visits': [website_visits],
        'cart_abandonment_rate': [cart_abandonment_rate],
        'product_preference': [product_preference],
        'device_type': [device_type],
        'newsletter_subscribed': [newsletter],
        'customer_reviews': [reviews],
        'last_purchase_days_ago': [last_purchase_days],
        'customer_lifetime_value': [total_spend / (tenure_months + 1)],
        'purchase_velocity': [total_purchases / (tenure_months + 1)],
        'recent_purchase': [1 if last_purchase_days <= 30 else 0],
        'review_engagement_rate': [reviews / (total_purchases + 1)],
        'engagement_score': [(website_visits / 100) + (reviews / 10)]
    })

    # Encode categorical variables
    input_encoded = input_data.copy()
    for col in categorical_cols:
        if col in input_encoded.columns:
            input_encoded[col] = label_encoders[col].transform(input_encoded[col])

    # Scale features
    input_scaled = best_scaler.transform(input_encoded)

    # Get prediction and probability
    prediction = best_model.predict(input_scaled)
    probabilities = best_model.predict_proba(input_scaled)

    # Get class label
    predicted_label = best_le_target.inverse_transform([prediction])

    # Get confidence
    confidence = max(probabilities) * 100

    # Create output message
    output = f"**Predicted Purchase Frequency:** {predicted_label}\n\n**Confidence:** {confidence:.1f}%\n\n"
    output += "**Probability Distribution:**\n"
    for i, label in enumerate(best_le_target.classes_):
        output += f"- {label}: {probabilities[i]*100:.1f}%\n"

    return output

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_purchase_frequency,
    inputs=[
        gr.Number(label="Age (years)", value=35),
        gr.Dropdown(['M', 'F'], label="Gender"),
        gr.Number(label="Total Purchases", value=20),
        gr.Number(label="Total Spend ($)", value=5000),
        gr.Number(label="Average Order Value ($)", value=250),
        gr.Number(label="Customer Tenure (months)", value=24),
        gr.Number(label="Website Visits", value=150),
        gr.Slider(0, 100, step=1, label="Cart Abandonment Rate (%)", value=15),
        gr.Dropdown(['Electronics', 'Fashion', 'Home', 'Beauty'],
                   label="Product Preference", value='Electronics'),
        gr.Dropdown(['Mobile', 'Desktop', 'Tablet'],
                   label="Device Type", value='Desktop'),
        gr.Dropdown(['Yes', 'No'], label="Newsletter Subscribed", value='Yes'),
        gr.Number(label="Customer Reviews", value=10),
        gr.Number(label="Last Purchase Days Ago", value=15)
    ],
    outputs=gr.Textbox(label="Prediction Result"),
    title="🛍️ Customer Purchase Frequency Predictor",
    description="Enter customer details to predict their purchase frequency category.",
    theme=gr.themes.Soft()
)

# Launch the app (uncomment to run)
# interface.launch()

print("\n✅ Model training and Gradio interface setup complete!")
print("To launch the app, uncomment the interface.launch() line at the end.")
