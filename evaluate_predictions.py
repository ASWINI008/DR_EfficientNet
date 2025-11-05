import csv

csv_path = r"C:\Users\aswin\Documents\DR_EfficientNet\predictions.csv"

correct = 0
total = 0

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        if row['Predicted_Class'] == row['True_Class']:
            correct += 1

accuracy = (correct / total) * 100
print(f"Test set accuracy: {accuracy:.2f}%")
