import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import dash
from dash import dcc, html, Input, Output

# Step 1: Load and Clean Data
file_path = 'hotel_bookings.csv'
df = pd.read_csv(file_path)
df.dropna(inplace=True)
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
df = df[df['lead_time'] >= 0]

# Step 2: Machine Learning Model
features = ['lead_time', 'previous_cancellations', 'total_of_special_requests', 'booking_changes']
X = df[features]
y = df['is_canceled']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Step 3: Prepare Dashboard
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1('Hotel Booking Cancellation Dashboard', style={'textAlign': 'center'}),
    html.Div(f"Model Accuracy: {accuracy:.2%}", style={'textAlign': 'center', 'marginBottom': '20px'}),
    dcc.Graph(figure=px.histogram(df, x='is_canceled', title='Booking Cancellations')),
    dcc.Graph(figure=px.histogram(df, x='hotel', color='is_canceled', title='Cancellations by Hotel Type')),
    dcc.Graph(figure=px.histogram(df, x='arrival_date_month', color='is_canceled', title='Monthly Booking Trends')),
    dcc.Graph(figure=px.scatter(df, x='lead_time', y='is_canceled', title='Lead Time vs Cancellations')),
    dcc.Graph(figure=px.bar(df, x='total_of_special_requests', y='is_canceled', title='Special Requests vs Cancellations')),
])

if __name__ == '__main__':
    app.run_server(debug=True)