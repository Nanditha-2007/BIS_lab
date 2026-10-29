
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

np.random.seed(42)

# 1. Load the dataset
data = load_breast_cancer()
X = data.data
y = data.target

# 2. Split data into training, validation and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=42, stratify=y_temp
)

# Normalize features using training data only
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 3. Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Calculate class-wise log likelihoods for each feature
def feature_scores(X):
    scores = []

    for c in model.classes_:
        mean = model.theta_[list(model.classes_).index(c)]
        var = model.var_[list(model.classes_).index(c)]

        log_prob = -0.5 * (
            np.log(2 * np.pi * var)
            + ((X[:, :, None] - mean) ** 2 / var)
        )

        scores.append(log_prob[:, :, 0])

    return np.stack(scores, axis=1)

# Feature-wise log likelihoods for validation and test data
val_scores = feature_scores(X_val)
test_scores = feature_scores(X_test)

# Include class prior probabilities
log_prior = np.log(model.class_prior_)

# 4. Weighted Naive Bayes classification
def predict_weighted(scores, weights):
    weighted_scores = np.sum(
        scores * weights[None, None, :], axis=2
    )
    weighted_scores += log_prior[None, :]
    return model.classes_[np.argmax(weighted_scores, axis=1)]

def fitness(weights):
    predictions = predict_weighted(val_scores, weights)
    return accuracy_score(y_val, predictions)

# 5. Cuckoo Search parameters
NESTS = 10
ITERATIONS = 50
PA = 0.25
DIM = X_train.shape[1]

# Generate initial nests (feature weights)
nests = np.random.uniform(0, 2, (NESTS, DIM))
scores = np.array([fitness(nest) for nest in nests])

best_index = np.argmax(scores)
best_nest = nests[best_index].copy()
best_fitness = scores[best_index]
history = []

# 6. Cuckoo Search optimization
for iteration in range(ITERATIONS):

    for i in range(NESTS):
        # Levy-flight-inspired step using a Cauchy distribution
        step = np.random.standard_cauchy(DIM) * 0.01
        new_nest = np.clip(nests[i] + step, 0, 2)
        new_fitness = fitness(new_nest)

        j = np.random.randint(NESTS)

        if new_fitness > scores[j]:
            nests[j] = new_nest
            scores[j] = new_fitness

    # Abandon some nests and generate new solutions
    for i in range(NESTS):
        if np.random.rand() < PA:
            nests[i] = np.random.uniform(0, 2, DIM)
            scores[i] = fitness(nests[i])

    # Keep track of the best solution found
    best_index = np.argmax(scores)

    if scores[best_index] > best_fitness:
        best_fitness = scores[best_index]
        best_nest = nests[best_index].copy()

    history.append(best_fitness)

# 7. Evaluate the selected weights on unseen test data
test_predictions = predict_weighted(test_scores, best_nest)
test_accuracy = accuracy_score(y_test, test_predictions)

print("Cuckoo Search - Pattern Recognition")
print("-----------------------------------")
print("Dataset: Breast Cancer Wisconsin")
print("Number of features:", DIM)
print("Best validation accuracy:",
      round(best_fitness * 100, 2), "%")
print("Test accuracy:",
      round(test_accuracy * 100, 2), "%")
print("Optimized feature weights:")
print(np.round(best_nest, 3))

# 8. Plot convergence
plt.plot(range(1, ITERATIONS + 1), history, marker="o")
plt.xlabel("Iteration")
plt.ylabel("Best Validation Accuracy")
plt.title("Cuckoo Search Convergence")
plt.grid(True)
plt.show()
 