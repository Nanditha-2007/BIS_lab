import random
import math
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PARAMETERS
# --------------------------------------------------

POPULATION_SIZE = 100
GENERATIONS = 30
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.3

START = (0, 0)
GOAL = (10, 10)

# Obstacles: (xmin, ymin, xmax, ymax)
OBSTACLES = [
    (3, 2, 5, 7),
    (6, 5, 8, 9)
]

# Number of intermediate points in a path
NUM_POINTS = 5


# --------------------------------------------------
# 2. CREATE A RANDOM PATH
# --------------------------------------------------

def create_path():

    path = [START]

    for _ in range(NUM_POINTS):

        x = random.uniform(0, 10)
        y = random.uniform(0, 10)

        path.append((x, y))

    path.append(GOAL)

    return path


# --------------------------------------------------
# 3. CREATE INITIAL POPULATION
# --------------------------------------------------

def create_population():

    population = []

    for _ in range(POPULATION_SIZE):
        population.append(create_path())

    return population


# --------------------------------------------------
# 4. CHECK IF A POINT IS INSIDE AN OBSTACLE
# --------------------------------------------------

def point_in_obstacle(point):

    x, y = point

    for xmin, ymin, xmax, ymax in OBSTACLES:

        if xmin <= x <= xmax and ymin <= y <= ymax:
            return True

    return False


# --------------------------------------------------
# 5. CHECK IF PATH IS VALID
# --------------------------------------------------

def path_is_valid(path):

    for point in path:

        if point_in_obstacle(point):
            return False

    return True


# --------------------------------------------------
# 6. CALCULATE PATH LENGTH
# --------------------------------------------------

def path_length(path):

    distance = 0

    for i in range(len(path) - 1):

        x1, y1 = path[i]
        x2, y2 = path[i + 1]

        distance += math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2
        )

    return distance


# --------------------------------------------------
# 7. FITNESS FUNCTION
# --------------------------------------------------

def fitness(path):

    # Invalid path gets zero fitness
    if not path_is_valid(path):
        return 0

    distance = path_length(path)

    # Shorter path = higher fitness
    return 1 / (distance + 1)


# --------------------------------------------------
# 8. ROULETTE-WHEEL SELECTION
# --------------------------------------------------

def selection(population):

    fitness_values = [fitness(path) for path in population]

    total_fitness = sum(fitness_values)

    # If every path is invalid
    if total_fitness == 0:
        return random.choice(population)

    probabilities = [
        f / total_fitness
        for f in fitness_values
    ]

    return random.choices(
        population,
        weights=probabilities,
        k=1
    )[0]


# --------------------------------------------------
# 9. CROSSOVER
# --------------------------------------------------

def crossover(parent1, parent2):

    if random.random() > CROSSOVER_RATE:
        return parent1.copy(), parent2.copy()

    # Select crossover point
    point = random.randint(1, NUM_POINTS)

    child1 = parent1[:point] + parent2[point:]

    child2 = parent2[:point] + parent1[point:]

    return child1, child2


# --------------------------------------------------
# 10. MUTATION
# --------------------------------------------------

def mutation(path):

    mutated_path = path.copy()

    # Don't mutate START and GOAL
    for i in range(1, len(mutated_path) - 1):

        if random.random() < MUTATION_RATE:

            x = random.uniform(0, 10)
            y = random.uniform(0, 10)

            mutated_path[i] = (x, y)

    return mutated_path


# --------------------------------------------------
# 11. GENETIC ALGORITHM
# --------------------------------------------------

def genetic_algorithm():

    # Initial population
    population = create_population()

    best_path = None
    best_fitness = 0

    # Repeat for generations
    for generation in range(GENERATIONS):

        # Find best path in current population
        for path in population:

            current_fitness = fitness(path)

            if current_fitness > best_fitness:

                best_fitness = current_fitness
                best_path = path.copy()

        print(
            "Generation:",
            generation + 1,
            "Best Fitness:",
            round(best_fitness, 6)
        )

        # Create next generation
        new_population = []

        while len(new_population) < POPULATION_SIZE:

            # Selection
            parent1 = selection(population)
            parent2 = selection(population)

            # Crossover
            child1, child2 = crossover(
                parent1,
                parent2
            )

            # Mutation
            child1 = mutation(child1)
            child2 = mutation(child2)

            new_population.append(child1)

            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

    return best_path


# --------------------------------------------------
# 12. RUN THE GENETIC ALGORITHM
# --------------------------------------------------

best_path = genetic_algorithm()

print("\nBest Path:")

for point in best_path:

    print(
        "(",
        round(point[0], 2),
        ",",
        round(point[1], 2),
        ")"
    )

print(
    "\nPath Length:",
    round(path_length(best_path), 2)
)

print(
    "Fitness:",
    round(fitness(best_path), 6)
)


# --------------------------------------------------
# 13. VISUALIZE THE BEST PATH
# --------------------------------------------------

x_values = [point[0] for point in best_path]
y_values = [point[1] for point in best_path]

plt.figure(figsize=(8, 8))

# Draw obstacles
for xmin, ymin, xmax, ymax in OBSTACLES:

    rectangle = plt.Rectangle(
        (xmin, ymin),
        xmax - xmin,
        ymax - ymin,
        fill=True
    )

    plt.gca().add_patch(rectangle)


# Draw robot path
plt.plot(
    x_values,
    y_values,
    marker='o',
    linewidth=2
)

# Start point
plt.scatter(
    START[0],
    START[1],
    s=100,
    label="Start"
)

# Goal point
plt.scatter(
    GOAL[0],
    GOAL[1],
    s=100,
    label="Goal"
)

plt.xlim(0, 10)
plt.ylim(0, 10)

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "Genetic Algorithm - Mobile Robot Path Planning"
)

plt.legend()
plt.grid()

plt.show()
