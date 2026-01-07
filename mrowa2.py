import numpy as np

class AntColonyOptimization:
    def __init__(self, dist_matrix, orders, base_node, params):
        #Params
        self.iterations = params[0] # Number of iterations
        self.ants       = params[1] # Number of ants
        self.alpha      = params[2] # Pheromone weight
        self.beta       = params[3] # Distance weight
        self.rho        = params[4] # Evaporation rate

        self.base_node = base_node            # Base set
        self.orders = orders                  # List of orders
        self.cities = dist_matrix.shape[0]    # Number of cities
        
        # Initialize the distance matrix by calculating shortest paths between all nodes
        self.dist_matrix, self.next_node = self._floyd_warshall_with_path(dist_matrix)

        self.pheromone = np.ones((self.cities, self.cities)) * 0.1          # Initializing pheromones
        
        self.history_best_dist = []                                         # List using to show how distance decrease in every iteration
        self.global_best_path = None                                        # Shortest path
        self.global_best_dist = float('inf')                                # Distance the shortest path
        self.orders_sequence_history = None                               # Best sequence of order indices

    def _floyd_warshall_with_path(self, matrix):
        n = self.cities                                    # Number cities
        dist = np.full((n, n), float('inf'))               # Matrix with inf
        next_node = np.full((n, n), -1, dtype=int)         # Matrix with int(-1), next node on path

        for i in range(n):
            for j in range(n):
                if i == j:
                    dist[i][j] = 0                         # Put 0 on diagonal
                elif matrix[i][j] is not None:
                    dist[i][j] = float(matrix[i][j])
                    next_node[i][j] = j

        # Floyd Warshall Heart
        for k in range(n): 
            for i in range(n):
                for j in range(n):
                    if dist[i][j] > dist[i][k] + dist[k][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
                        next_node[i][j] = next_node[i][k]
        
        dist[dist == float('inf')] = 1e9 # Change inf to vary big number
        return dist, next_node

    def _get_full_path_(self, u, v):
        if self.next_node[u][v] == -1: 
            return [u, v]
        
        full_path = [u]
        temp_u = u

        # We use next_node matrix to create full path
        while temp_u != v:
            temp_u = self.next_node[temp_u][v]
            if temp_u == -1: 
                break
            full_path.append(temp_u)
        return full_path

    def _get_move_probability(self, current_node, allowed_nodes):
        probabilities = []
        for next_node in allowed_nodes:
            # We check pheromone between current node and the target pickup node
            p_val = self.pheromone[current_node, next_node]                             # Pheromon value
            d_val = self.dist_matrix[current_node, next_node]                           # Distance value
            
            if d_val >= 1e9:                                                            # Security for forbidden connection
                probabilities.append(0)
                continue

            tau = p_val ** self.alpha                                                   # Pheromone trail intensity
            # Raising (1/dist) to the power of Beta can create very small numbers; we cap it for stability
            eta = (1.0 / (d_val + 1e-6)) ** self.beta                                   # Heuristic desirability 
            probabilities.append(tau * eta)                                             # Product of tau end eta
        
        total = sum(probabilities)                                                      # Sum of probabilities
        if total <= 1e-12:                                                              # If all paths have zero probability or underflow
            return [1.0 / len(allowed_nodes)] * len(allowed_nodes)                      # Random choose 
            
        return [p / total for p in probabilities]                                       # Normalize values for probabilities

    def solve(self):
        for _ in range(self.iterations):
            all_paths = []                                                              # List with all paths in history
            all_distances = []                                                          # List with distances in every path
            iteration_order_sequences = []                                              # Order sequences in current iteration
            
            for _ in range(self.ants):
                path, dist, order_indices = self._run_ant()                             # Start simulation with all ants
                all_paths.append(path)                                                  # Save path
                all_distances.append(dist)                                              # Save distance
                iteration_order_sequences.append(order_indices)
                
                if dist < self.global_best_dist:                                        # Chosse the shortest path
                    self.global_best_dist = dist
                    self.global_best_path = path
                    self.orders_sequence_history = order_indices                       # Save best order sequence
            
            self.history_best_dist.append(min(all_distances))                           # Save the smallest distance in every iteration
            
            # PHEROMONE UPDATE LOGIC
            self.pheromone *= (1 - self.rho)                                            # Pheromone evaporation
            
            # ELITIST STRATEGY: Only the top ants reinforce the paths to reduce noise
            combined = list(zip(all_distances, iteration_order_sequences))
            combined.sort(key=lambda x: x[0])  # Sort by distance (ascending)
            
            for dist, order_seq in combined[:max(1, self.ants // 4)]:                   # Update for top 25% of ants
                if 0 < dist < 1e9:
                    pheromone_value = 100.0 / dist                                      # Q=100 scaling factor
                    
                    curr = self.base_node
                    for o_idx in order_seq:
                        p_node, d_node, _ = self.orders[o_idx]
                        
                        # Reinforce the decision points: current -> pickup AND pickup -> delivery
                        self.pheromone[curr, p_node] += pheromone_value
                        self.pheromone[p_node, d_node] += pheromone_value
                        curr = d_node
                    
                    # Reinforce the return to base
                    self.pheromone[curr, self.base_node] += pheromone_value
                        
        return self.global_best_path, self.global_best_dist, self.history_best_dist, self.orders_sequence_history

    def _run_ant(self):
        current_node = self.base_node                       # Start in base
        path = [current_node]                               # On the beginning path has only one city
        total_dist = 0
        order_sequence = []                                 # List to store order execution sequence
        
        remaining_orders = list(range(len(self.orders)))    # List with number orders

        while remaining_orders:
            allowed_orders_indices = list(range(len(remaining_orders)))                                 # List index
            potential_pickups = [self.orders[remaining_orders[i]][0] for i in allowed_orders_indices]   # List with cities, where we need take a order
            
            probs = self._get_move_probability(current_node, potential_pickups)                         # List with probabilities, it helps to take a decision where to go in next move
            
            local_index = np.random.choice(allowed_orders_indices, p=probs)                             # Choice index from list with weights
            order_index = remaining_orders.pop(local_index)                                             # Assign the actual order and delete index where we picked up a order
            order_sequence.append(order_index)                                                          # Save order index to sequence
            
            p_node, d_node, _ = self.orders[order_index]                                                # Assign pick_up and delivery node from order_index
            
            total_dist += self.dist_matrix[current_node, p_node]                                        # Add distance from current node to pick_up node
            segment_p = self._get_full_path_(current_node, p_node)
            path.extend(segment_p[1:])           

            total_dist += self.dist_matrix[p_node, d_node]                                              # Add distance from pick_up node to delivery node
            segment_d = self._get_full_path_(p_node, d_node)
            path.extend(segment_d[1:])
            
            current_node = d_node                                                                       # Change current node to delivery node
            
        if current_node != self.base_node:                                                              # Security, where the end of order is in base
            total_dist += self.dist_matrix[current_node, self.base_node]                                # Comeback to base
            segment_b = self._get_full_path_(current_node, self.base_node)
            path.extend(segment_b[1:])

        cleaned_path = [path[0]]                                                                        # Create new list to delete repetitive nodes next to each other
        for node in path[1:]:
            if node != cleaned_path[-1]:
                cleaned_path.append(node)
        path = cleaned_path

        return path, total_dist, order_sequence

# --- ZMIANY W PARAMETRACH DLA WIDOCZNEJ ZBIEŻNOŚCI ---
# Przy 40 miastach Beta=4.0 jest OK, ale Alfa powinna być wyższa, by mrówki słuchały feromonów.
# Rho=0.1 pozwala feromonom trwać wystarczająco długo, by kolonia się uczyła.


params = (150, 60, 1.0, 4.0, 0.1)

np.random.seed(42)
size = 40
dist_m = np.random.randint(10, 101, size=(size, size)).astype(object)

for i in range(size):
    for j in range(i, size):
        if i == j:
            dist_m[i, j] = None
        else:
            # Szansa 30% na brak bezpośredniego połączenia (None)
            if np.random.rand() < 0.3:
                dist_m[i, j] = None
                dist_m[j, i] = None
            else:
                dist_m[j, i] = dist_m[i, j]
orders = [
    (0, 15, 100), (4, 19, 50), (12, 1, 80), (7, 3, 120), (18, 5, 60),
    (22, 35, 90), (39, 10, 110), (25, 8, 70), (33, 2, 130), (11, 28, 85),
    (30, 5, 95), (14, 38, 120), (2, 21, 65), (36, 17, 105), (9, 31, 75),
    (20, 6, 115), (13, 27, 80), (37, 3, 140), (1, 24, 55), (29, 32, 90)
]
base = 4

aco = AntColonyOptimization(dist_m, orders, base, params)    # Create simulation
best_path, best_dist, history, orders_sequence = aco.solve() # Start simulation

best_path = [int(x) for x in best_path] 
history = [int(x) for x in history]

