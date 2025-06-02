import csv
import matplotlib.pyplot as plt

# Load data from CSV
def load_data(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        header = next(reader)
        expected_len = len(header)
        data = []
        for row in reader:
            if len(row) != expected_len:
                continue
            try:
                data.append(list(map(float, row)))
            except ValueError:
                continue
        return data, header

# Split dataset into train and test
def split_data(data, test_ratio=0.3):
    split_point = int(len(data) * (1 - test_ratio))
    return data[:split_point], data[split_point:]

# Simple decision tree rules (manual)
def simple_decision_tree(features):
    if len(features) < 6:
        return 0
    if features[2] < 130:
        return 1
    elif features[3] > 230:
        return 0
    elif features[5] > 150:
        return 0
    else:
        return 1

# Limited depth tree
def limited_depth_tree(features):
    if len(features) < 8:
        return 0
    if features[0] > 50:
        if features[7] > 1:
            return 0
        else:
            return 1
    else:
        if features[4] < 200:
            return 1
        else:
            return 0

# Simulated Random Forest (ensemble of weak rules)
def random_forest_classifier(features):
    if len(features) < 10:
        return 0
    votes = []
    votes.append(1 if features[2] < 130 else 0)
    votes.append(1 if features[0] < 50 else 0)
    votes.append(1 if features[9] < 0.5 else 0)
    votes.append(1 if features[4] < 200 else 0)
    return 1 if sum(votes) >= 3 else 0

# Accuracy calculator
def calculate_accuracy(data, model):
    correct = 0
    for row in data:
        features = row[:-1]
        label = row[-1]
        prediction = model(features)
        if prediction == label:
            correct += 1
    return correct / len(data)

# Visualization (just for title barplot to simulate feature importance)
def draw_bar_plot():
    features = ['chol', 'age', 'thalach', 'oldpeak']
    importances = [0.3, 0.25, 0.25, 0.2]
    plt.figure(figsize=(8, 5))
    plt.barh(features, importances, color='skyblue')
    plt.title("Simulated Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_simulated.png")
    plt.show()

# Run the program
if __name__ == "__main__":
    data, header = load_data("heart.csv")
    train_data, test_data = split_data(data)

    acc_dt = calculate_accuracy(test_data, simple_decision_tree)
    acc_limited = calculate_accuracy(test_data, limited_depth_tree)
    acc_rf = calculate_accuracy(test_data, random_forest_classifier)

    print("Simple Decision Tree Accuracy:", acc_dt)
    print("Limited Depth Tree Accuracy:", acc_limited)
    print("Random Forest-like Accuracy:", acc_rf)

    draw_bar_plot()
