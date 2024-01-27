from flask import Flask, render_template, request
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pytesseract
import os
import pandas as pd

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

app = Flask(__name__)

# Function to extract text from the uploaded receipt
def extract_text(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Function to calculate environmental impact based on the number of items
def calculate_environmental_impact(text):
    # A simple example: 1 item contributes 5 units of environmental impact
    items = text.split('\n')  # Assuming each line is an item
    environmental_impact = len(items) * 5
    return environmental_impact

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to convert quantity strings to numeric values
def parse_quantity(quantity_str):
    # Placeholder function, replace with your implementation
    return float(quantity_str.split()[0])

# Function to train and predict carbon footprint using linear regression
def predict_carbon_footprint(train_data):
    X = train_data[['Quantity']]
    y = train_data['CarbonFootprint']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Create a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy (for demonstration purposes)
    accuracy = metrics.r2_score(y_test, y_pred)

    return model, accuracy

# Route to display the index page
@app.route('/')
def index():
    return render_template('index.html')

# Route to upload a receipt
@app.route('/upload', methods=['POST'])
def upload():
    if 'receipt' not in request.files:
        return 'No file part'

    file = request.files['receipt']
    if file.filename == '':
        return 'No selected file'

    # Save the uploaded file
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Extract text from the receipt
    receipt_text = extract_text(file_path)

    # Calculate environmental impact
    environmental_impact = calculate_environmental_impact(receipt_text)

    return render_template('result.html', result_text=receipt_text, environmental_impact=environmental_impact)

# Route to analyze and predict carbon footprint
@app.route('/analyze', methods=['GET'])
def analyze():
    # Read data from the CSV file (adjust the file path)
    data = pd.read_csv('Food_Production.csv')

    # Convert quantity strings to numeric values (if needed)
    data['Quantity'] = data['Quantity'].apply(parse_quantity)

    # Train the machine learning model
    model, accuracy = predict_carbon_footprint(data)

    # Make predictions on the entire dataset
    X_all = data[['Quantity']]
    predictions = model.predict(X_all)

    # Add predictions to the data DataFrame
    data['PredictedCarbonFootprint'] = predictions

    # Return the result to the client
    return render_template('analysis_result.html', grocery_items=data.to_dict('records'), accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)



