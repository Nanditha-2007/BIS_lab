import random
import math
import matplotlib.pyplot as plt

# -----------------------------------------
# PSO PARAMETERS
# -----------------------------------------
NUM_PARTICLES = 20
ITERATIONS = 30

W = 0.7       # Inertia weight
C1 = 1.5      # Personal learning factor
C2 = 1.5      # Social learning factor

NUM_JOINTS = 6

# Desired configuration of the medical robot
TARGET = [30, 45, 60, 40, 50, 35]

# Allowed joint angle range
MIN_ANGLE = 0
MAX_ANGLE = 90


# -----------------------------------------
# FITNESS FUNCTION
# -----------------------------------------
# Lower fitness = better solution
def fitness(position):

    total = 0

    for i in range(NUM_JOINTS):
        difference = position[i] - TARGET[i]
        total += difference ** 2

    return math.sqrt(total)


# -----------------------------------------
# INITIALIZE PARTICLES
# -----------------------------------------
particles = []

for i in range(NUM_PARTICLES):

    # Random joint angles
    position = [
        random.uniform(MIN_ANGLE, MAX_ANGLE)
        for _ in range(NUM_JOINTS)
    ]

    # Random velocities
    velocity = [
        random.uniform(-5, 5)
        for _ in range(NUM_JOINTS)
    ]

    particle = {
        "position": position,
        "velocity": velocity,

        # Initially current position is pBest
        "pbest": position.copy(),
        "pbest_fitness": fitness(position)
    }

    particles.append(particle)


# -----------------------------------------
# FIND INITIAL GLOBAL BEST
# -----------------------------------------
gbest = particles[0]["pbest"].copy()
gbest_fitness = particles[0]["pbest_fitness"]

for particle in particles:

    if particle["pbest_fitness"] < gbest_fitness:

        gbest = particle["pbest"].copy()
        gbest_fitness = particle["pbest_fitness"]


# Store best fitness for graph
best_fitness_history = []


# -----------------------------------------
# PSO ITERATIONS
# -----------------------------------------
for iteration in range(ITERATIONS):

    for particle in particles:

        position = particle["position"]
        velocity = particle["velocity"]

        # Random values
        r1 = random.random()
        r2 = random.random()

        # ---------------------------------
        # UPDATE VELOCITY
        # ---------------------------------
        for j in range(NUM_JOINTS):

            velocity[j] = (
                W * velocity[j]
                + C1 * r1 *
                (particle["pbest"][j] - position[j])
                + C2 * r2 *
                (gbest[j] - position[j])
            )

        # ---------------------------------
        # UPDATE POSITION
        # ---------------------------------
        for j in range(NUM_JOINTS):

            position[j] += velocity[j]

            # Keep angle within valid range
            position[j] = max(
                MIN_ANGLE,
                min(MAX_ANGLE, position[j])
            )

        # ---------------------------------
        # CALCULATE FITNESS
        # ---------------------------------
        current_fitness = fitness(position)

        # ---------------------------------
        # UPDATE PERSONAL BEST
        # ---------------------------------
        if current_fitness < particle["pbest_fitness"]:

            particle["pbest"] = position.copy()
            particle["pbest_fitness"] = current_fitness

        # ---------------------------------
        # UPDATE GLOBAL BEST
        # ---------------------------------
        if current_fitness < gbest_fitness:

            gbest = position.copy()
            gbest_fitness = current_fitness

    # Store best fitness
    best_fitness_history.append(gbest_fitness)

    print(
        "Iteration:",
        iteration + 1,
        "Best Fitness:",
        f"{gbest_fitness:.4f}"
    )


# -----------------------------------------
# FINAL RESULT
# -----------------------------------------
print("\n===================================")
print("FINAL RESULT")
print("===================================")

print("\nTarget Joint Configuration:")
print([round(x, 2) for x in TARGET])

print("\nOptimized Joint Configuration:")
print([round(x, 2) for x in gbest])

print("\nFinal Fitness:")
print(f"{gbest_fitness:.4f}")


# -----------------------------------------
# GRAPH
# -----------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, ITERATIONS + 1),
    best_fitness_history,
    marker='o'
)

plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.title("PSO Optimization of Medical Robot Joint Configuration")

plt.grid()
plt.show()
