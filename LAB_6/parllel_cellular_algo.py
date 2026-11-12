
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

# --------------------------------
# 1. INITIALIZE PARAMETERS
# --------------------------------

W = 7                  # Number of CA cells
S = 8                  # Number of cell states
T = 7                  # Developmental steps

POP_SIZE = 50          # GA population
GENERATIONS = 50       # Reduced for practical execution
NOISE_LEVEL = 0.20     # 20% salt-and-pepper noise
TRAIN_SAMPLES = 150

random.seed(42)
np.random.seed(42)


# --------------------------------
# 2. FUNCTIONAL BLOCKS
# --------------------------------

def apply_function(code, a, b):

    if code == 0:
        return a

    elif code == 1:
        return b

    elif code == 2:
        return min(a, b)

    elif code == 3:
        return max(a, b)

    elif code == 4:
        return 255 - a

    elif code == 5:
        return a / 2

    elif code == 6:
        return b / 4

    elif code == 7:
        return (a + b) % 256

    elif code == 8:
        return min(255, a + b)

    elif code == 9:
        return (a + b) / 2

    else:
        return abs(a - b)


# --------------------------------
# 3. CREATE CHROMOSOME
# --------------------------------

def create_chromosome():

    # Initial states of CA cells
    initial = np.random.randint(0, S, W)

    # Total transition rules = S^3
    rules = np.zeros((S**3, 4), dtype=int)

    # Each rule contains:
    # Next state, function, input1, input2

    rules[:, 0] = np.random.randint(0, S, S**3)
    rules[:, 1] = np.random.randint(0, 11, S**3)
    rules[:, 2] = np.random.randint(0, 256, S**3)
    rules[:, 3] = np.random.randint(0, 256, S**3)

    return np.concatenate((initial, rules.flatten()))


# --------------------------------
# 4. CELLULAR AUTOMATON FILTER
# --------------------------------

def filter_pixel(chromosome, patch):

    initial = chromosome[:W]

    rules = chromosome[W:].reshape(S**3, 4)

    states = initial.copy()

    # First step uses 9 pixels of 3x3 window
    values = patch.flatten().astype(float)

    # Perform CA development
    for step in range(T):

        new_states = np.zeros(W, dtype=int)
        outputs = np.zeros(W, dtype=float)

        # Update all cells synchronously
        for i in range(W):

            left = states[(i - 1) % W]
            center = states[i]
            right = states[(i + 1) % W]

            # Find transition rule
            rule_index = left * S**2 + center * S + right

            next_state, func, i1, i2 = rules[rule_index]

            # Connect functional block inputs
            index1 = i1 % len(values)
            index2 = i2 % len(values)

            a = values[index1]
            b = values[index2]

            # Generate functional block output
            outputs[i] = apply_function(func, a, b)

            # Store next state
            new_states[i] = next_state

        # Replace states and block outputs
        states = new_states
        values = outputs

    # Central cell output is filtered pixel
    return np.clip(values[W // 2], 0, 255)


# --------------------------------
# 5. ADD SALT-AND-PEPPER NOISE
# --------------------------------

def add_noise(image):

    noisy = image.copy().astype(float)

    rows, cols = image.shape

    mask = np.random.random((rows, cols))

    noisy[mask < NOISE_LEVEL / 2] = 0

    noisy[mask > 1 - NOISE_LEVEL / 2] = 255

    return noisy.astype(np.uint8)


# --------------------------------
# 6. PREPARE TRAINING DATA
# --------------------------------

def get_training_data(original, noisy):

    rows, cols = original.shape

    data = []

    for _ in range(TRAIN_SAMPLES):

        i = random.randint(1, rows - 2)
        j = random.randint(1, cols - 2)

        patch = noisy[i-1:i+2, j-1:j+2]

        target = original[i, j]

        data.append((patch.copy(), target))

    return data


# --------------------------------
# 7. FITNESS USING MAE
# --------------------------------

def fitness(chromosome, data):

    error = 0

    for patch, target in data:

        output = filter_pixel(chromosome, patch)

        error += abs(output - target)

    return error / len(data)


# --------------------------------
# 8. TOURNAMENT SELECTION
# --------------------------------

def selection(population, scores):

    candidates = random.sample(range(POP_SIZE), 4)

    best = min(candidates, key=lambda x: scores[x])

    return population[best].copy()


# --------------------------------
# 9. MUTATION
# --------------------------------

def mutation(chromosome):

    child = chromosome.copy()

    # Select 6 candidate genes
    positions = random.sample(range(len(child)), 6)

    for index in positions:

        if random.random() < 0.96:

            # Initial CA state
            if index < W:

                child[index] = random.randrange(S)

            else:

                field = (index - W) % 4

                if field == 0:
                    child[index] = random.randrange(S)

                elif field == 1:
                    child[index] = random.randrange(11)

                else:
                    child[index] = random.randrange(256)

    return child


# --------------------------------
# 10. APPLY FILTER TO FULL IMAGE
# --------------------------------

def apply_filter(image, chromosome):

    rows, cols = image.shape

    result = image.copy().astype(float)

    for i in range(1, rows - 1):

        for j in range(1, cols - 1):

            patch = image[i-1:i+2, j-1:j+2]

            result[i, j] = filter_pixel(chromosome, patch)

    return np.clip(result, 0, 255).astype(np.uint8)


# --------------------------------
# 11. MAIN PROGRAM
# --------------------------------

# Read image from same folder as program
folder = os.path.dirname(os.path.abspath(__file__))

image_path = os.path.join(folder, "input.png")

original = Image.open(image_path).convert("L")

original = original.resize((128, 128))

original = np.array(original)

# Generate noisy image
noisy = add_noise(original)

# Prepare training samples
data = get_training_data(original, noisy)

# Initialize GA population
population = [
    create_chromosome()
    for _ in range(POP_SIZE)
]

best_solution = None
best_fitness = float("inf")

print("Starting Genetic Algorithm...\n")


# --------------------------------
# 12. GENETIC ALGORITHM
# --------------------------------

for gen in range(GENERATIONS):

    scores = [
        fitness(ch, data)
        for ch in population
    ]

    current_best = np.argmin(scores)

    if scores[current_best] < best_fitness:

        best_fitness = scores[current_best]

        best_solution = population[current_best].copy()

    # Elitism: retain best chromosome
    new_population = [best_solution.copy()]

    while len(new_population) < POP_SIZE:

        parent = selection(population, scores)

        child = mutation(parent)

        new_population.append(child)

    population = new_population

    print(
        "Generation:", gen + 1,
        "| Best MAE:", round(best_fitness, 3)
    )


# --------------------------------
# 13. FILTER IMAGE
# --------------------------------

filtered = apply_filter(noisy, best_solution)

# Calculate full-image MAE
mae = np.mean(
    np.abs(
        filtered[1:-1, 1:-1].astype(float)
        - original[1:-1, 1:-1].astype(float)
    )
)

print("\nBest Training Fitness:", round(best_fitness, 3))
print("Final Image MAE:", round(mae, 3))


# --------------------------------
# 14. DISPLAY RESULTS
# --------------------------------

plt.figure(figsize=(12, 5))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap="gray", vmin=0, vmax=255)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy, cmap="gray", vmin=0, vmax=255)
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(filtered, cmap="gray", vmin=0, vmax=255)
plt.title("Filtered Image")
plt.axis("off")

plt.tight_layout()

plt.savefig(
    os.path.join(folder, "PCA_Filtered_Result.png")
)

plt.show()

# Save filtered image
Image.fromarray(filtered).save(
    os.path.join(folder, "filtered_image.png")
)

print("\nFiltered image saved successfully!")
