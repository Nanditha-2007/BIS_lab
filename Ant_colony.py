
import random
import math
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. TSP PROBLEM: 15 CITIES
# ---------------------------------------
random.seed(42)

N = 15
cities = [(random.uniform(0, 100),
           random.uniform(0, 100)) for _ in range(N)]

distance = [[0.0] * N for _ in range(N)]

for i in range(N):
    for j in range(N):
        x1, y1 = cities[i]
        x2, y2 = cities[j]
        distance[i][j] = math.hypot(x2 - x1, y2 - y1)

# ---------------------------------------
# 2. COMMON ACO PARAMETERS
# ---------------------------------------
ALPHA = 1.0
BETA = 2.0
Q = 100
ITERATIONS = 80
REPEATS = 3

COLONY_SIZES = [5, 10, 15]
RHOS = [0.1, 0.5, 0.9]

# ACS-specific parameters
Q0 = 0.8
PHI = 0.1

# ---------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------
def route_length(route):
    return sum(
        distance[route[i]][route[i + 1]]
        for i in range(len(route) - 1)
    )


def choose_next(current, unvisited, pheromone, method):
    candidates = list(unvisited)

    weights = [
        (pheromone[current][j] ** ALPHA)
        * ((1.0 / distance[current][j]) ** BETA)
        for j in candidates
    ]

    if method == "ACS" and random.random() < Q0:
        return candidates[weights.index(max(weights))]

    return random.choices(candidates, weights=weights, k=1)[0]


def construct_route(pheromone, method):
    start = random.randrange(N)
    route = [start]
    unvisited = set(range(N))
    unvisited.remove(start)

    while unvisited:
        current = route[-1]

        nxt = choose_next(
            current, unvisited, pheromone, method
        )

        route.append(nxt)
        unvisited.remove(nxt)

        # ACS local pheromone update
        if method == "ACS":
            pheromone[current][nxt] = (
                (1 - PHI) * pheromone[current][nxt]
                + PHI
            )
            pheromone[nxt][current] = pheromone[current][nxt]

    route.append(start)
    return route


def deposit_route(pheromone, route, amount):
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        pheromone[a][b] += amount
        pheromone[b][a] = pheromone[a][b]


# ---------------------------------------
# 4. ACO VARIANTS
# ---------------------------------------
def run_aco(method, ants, rho):
    tau0 = 1.0
    pheromone = [
        [tau0 if i != j else 0.0 for j in range(N)]
        for i in range(N)
    ]

    best_route = None
    best_length = float("inf")

    # Bounds used by Max-Min ACO
    tau_max = Q / max(rho, 0.01)
    tau_min = tau_max / (2 * N)

    for iteration in range(ITERATIONS):
        routes = []
        lengths = []

        # Each ant constructs a tour
        for ant in range(ants):
            route = construct_route(pheromone, method)
            length = route_length(route)

            routes.append(route)
            lengths.append(length)

            if length < best_length:
                best_length = length
                best_route = route.copy()

        # Evaporation
        for i in range(N):
            for j in range(N):
                if i != j:
                    pheromone[i][j] *= (1 - rho)

        iteration_best_index = lengths.index(min(lengths))
        iteration_best_route = routes[iteration_best_index]
        iteration_best_length = lengths[iteration_best_index]

        if method == "ACS":
            # Global update on the best-so-far route
            deposit_route(
                pheromone, best_route, rho / best_length
            )

        elif method == "Elitist":
            # All ants deposit pheromone
            for route, length in zip(routes, lengths):
                deposit_route(pheromone, route, Q / length)

            # Extra pheromone on best-so-far route
            deposit_route(
                pheromone, best_route, N * Q / best_length
            )

        elif method == "Max-Min":
            # Only iteration-best ant deposits pheromone
            deposit_route(
                pheromone,
                iteration_best_route,
                Q / iteration_best_length
            )

            # Bound pheromone levels
            for i in range(N):
                for j in range(N):
                    if i != j:
                        pheromone[i][j] = max(
                            tau_min,
                            min(tau_max, pheromone[i][j])
                        )

    return best_route, best_length


# ---------------------------------------
# 5. COMPARE COLONY SIZES AND RHO VALUES
# ---------------------------------------
methods = ["ACS", "Elitist", "Max-Min"]
results = {}

overall_best_route = None
overall_best_length = float("inf")

for rho in RHOS:
    for ants in COLONY_SIZES:
        for method in methods:

            trial_lengths = []
            saved_route = None

            for trial in range(REPEATS):
                route, length = run_aco(method, ants, rho)
                trial_lengths.append(length)

                if length < overall_best_length:
                    overall_best_length = length
                    overall_best_route = route.copy()

                if saved_route is None or length < saved_route[1]:
                    saved_route = (route, length)

            average = sum(trial_lengths) / len(trial_lengths)

            results[(method, ants, rho)] = average

            print(
                f"{method:8} | Ants={ants:2} | "
                f"Rho={rho:.1f} | "
                f"Average distance={average:.2f}"
            )

# ---------------------------------------
# 6. PLOT COMPARISON GRAPHS
# ---------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, rho in zip(axes, RHOS):
    for method in methods:
        values = [
            results[(method, ants, rho)]
            for ants in COLONY_SIZES
        ]

        ax.plot(
            COLONY_SIZES,
            values,
            marker="o",
            label=method
        )

    ax.set_title(f"Evaporation rate rho = {rho}")
    ax.set_xlabel("Number of Ants")
    ax.set_ylabel("Average Route Distance")
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.show()

# ---------------------------------------
# 7. PLOT THE BEST ROUTE FOUND
# ---------------------------------------
plt.figure(figsize=(7, 7))

for i, (x, y) in enumerate(cities):
    plt.scatter(x, y)
    plt.text(x + 1, y + 1, str(i + 1))

for i in range(len(overall_best_route) - 1):
    a = overall_best_route[i]
    b = overall_best_route[i + 1]

    x1, y1 = cities[a]
    x2, y2 = cities[b]

    plt.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->")
    )

print("\nBest route found:")
print(" -> ".join(str(i + 1) for i in overall_best_route))
print(f"Best distance: {overall_best_length:.2f}")

plt.title("Best TSP Route Found by ACO Variants")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()
