import random
import argparse
import matplotlib.pyplot as plt

class SingleCargoGA:
    def __init__(self, route_data, orders, 
                 pop_size=100, generations=200, mutation_rate=0.05):
        self.route_data = route_data
        self.orders = orders
        self.route_len = len(route_data)
     
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        
        self.cargo_status = self._simulate_cargo_on_route()
        
        loaded_orders = {status['order_id'] for status in self.cargo_status if status['order_id'] is not None}
        orders_dict = {i: o for i, o in enumerate(orders)}
        self.base_revenue = sum(orders_dict[oid][2] for oid in loaded_orders)

    def _simulate_cargo_on_route(self):
        """mapowanie gdzie jest ładunek"""
        cargo_map = []
        current_order = None
        pending_orders = [list(o) + [i] for i, o in enumerate(self.orders)]
        
        for i, node_info in enumerate(self.route_data):
            node_id = node_info[0]
            step_info = {'value': 0, 'order_id': None, 'action': 'empty'}
            
            # Rozładunek
            if current_order and current_order[1] == node_id:
                current_order = None
                step_info['action'] = 'unload'
            
            # Załadunek
            if current_order is None:
                found_idx = -1
                for idx, order in enumerate(pending_orders):
                    start, end, profit, original_idx = order
                    if start == node_id:
                        current_order = (original_idx, end, profit)
                        found_idx = idx
                        step_info['action'] = 'load'
                        step_info['value'] = profit
                        step_info['order_id'] = original_idx
                        break
                if found_idx != -1:
                    pending_orders.pop(found_idx)
            
            # Transport
            elif current_order is not None:
                step_info['value'] = current_order[2]
                step_info['order_id'] = current_order[0]
                step_info['action'] = 'carry'
            
            cargo_map.append(step_info)
        return cargo_map

    def fitness(self, chromosome):
        penalty = 0
        for i in range(self.route_len):
            _, prob_robbery, cost_security = self.route_data[i]
            cargo_val = self.cargo_status[i]['value']
            buy_security = chromosome[i]
            
            if buy_security == 1:
                penalty += cost_security
            else:
                expected_loss = cargo_val * prob_robbery
                penalty += expected_loss
        return self.base_revenue - penalty

    def create_individual(self):
        return [random.randint(0, 1) for _ in range(self.route_len)]

    def crossover(self, p1, p2):
        pt = random.randint(1, self.route_len - 1)
        return p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]

    def mutate(self, ind):
        for i in range(len(ind)):
            if random.random() < self.mutation_rate:
                ind[i] = 1 - ind[i]
        return ind

    def run(self):
        population = [self.create_individual() for _ in range(self.pop_size)]
        history_best = []
        
        best_sol = None
        best_fit_overall = -float('inf')

        for _ in range(self.generations):
            fits = [self.fitness(ind) for ind in population]
            
            current_max = max(fits)

            if current_max > best_fit_overall:
                best_fit_overall = current_max
                best_sol = population[fits.index(current_max)]
            
            history_best.append(best_fit_overall)
    
            
            new_pop = [best_sol] 
            while len(new_pop) < self.pop_size:
                p1 = random.choice(population)
                p2 = random.choice(population)
                parent_a = p1 if self.fitness(p1) > self.fitness(p2) else p2
                
                p3 = random.choice(population)
                p4 = random.choice(population)
                parent_b = p3 if self.fitness(p3) > self.fitness(p4) else p4
                
                c1, c2 = self.crossover(parent_a, parent_b)
                new_pop.append(self.mutate(c1))
                if len(new_pop) < self.pop_size:
                    new_pop.append(self.mutate(c2))
            population = new_pop

        return best_sol, history_best



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="GA Optymalizacja Ochrony")
    parser.add_argument('--pop_size', type=int, default=50, help='Rozmiar populacji')
    parser.add_argument('--gen', type=int, default=100, help='Liczba generacji')
    parser.add_argument('--mut', type=float, default=0.05, help='Szansa mutacji')
    
    args = parser.parse_args()

    trasa_input = [
        (0, 0.10, 50),
        (1, 0.50, 200),
        (2, 0.10, 50),
        (3, 0.05, 20),
        (4, 0.01, 10),
        (5, 0.01, 1000), 
        (4, 0.05, 10),
        (3, 0.1, 20), 
        (2, 0.5, 120),
        (1, 0.1, 220),
        (10, 0.99, 4),
        (11, 0.2, 40),
        (12, 0.3, 45),
        (13, 0.3, 400),
        (14, 0.7, 2000),
        (11, 0.01, 4),
        (16, 0.02, 43),
        (17, 0.2, 1000),
        (18, 0.3, 400),
        (19, 0.2, 400),
        (2, 0.32, 800),
        (1, 0.42, 1200),   
        (0, 0.32, 5000),
    ]

    zamowienia_input = [
        (0, 2, 1000),
        (3, 4, 300),
        (5, 10, 200),
        (10, 13, 200),
        (14, 19, 300),
        (2, 1, 1000),
    ]
    
    ga = SingleCargoGA(trasa_input, zamowienia_input, 
                       pop_size=args.pop_size, generations=args.gen, mutation_rate=args.mut)
    
    best_chromosome, fit_history = ga.run()
    best_score = fit_history[-1]
    print(best_chromosome)
    # Wyświetlanie tabeli wyników
    print(f"\nStart z parametrami: Populacja={args.pop_size}, Generacje={args.gen}, Mutacja={args.mut}")
    print(f"Maksymalny możliwy przychód: {ga.base_revenue}")
    print(f"Osiągnięty zysk netto: {best_score:.2f}")
    
    print("\n" + "-"*65)
    print(f"{'Węzeł':<6} | {'Status':<8} | {'Ładunek':<8} | {'Ochrona':<8} | {'Decyzja'}")
    print("-" * 65)
    
    for i, gene in enumerate(best_chromosome):
        
        node_id = trasa_input[i][0]
        prob = trasa_input[i][1]
        cost = trasa_input[i][2]
        val = ga.cargo_status[i]['value']
        
        decyzja = "TAK" if gene else "NIE"
        ryzyko = val * prob
        
        info = ""
        if val == 0: 
            info = "Pusto (OK)"
        elif gene:
            if cost < ryzyko: info = "Zyskowna"
            else: info = "Nieopłacalna?"
        else:
            if cost > ryzyko: info = "Oszczędność"
            else: info = "Ryzykowna!"

        print(f"ID: {node_id:<2} | {ga.cargo_status[i]['action']:<8} | {val:<8} | {decyzja:<8} | {info}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(fit_history)), fit_history, label='Najlepszy wynik (Best-so-far)', color='red', linewidth=2)
    plt.title(f'Postęp Algorytmu Genetycznego (Pop: {args.pop_size}, Gen: {args.gen})')
    plt.xlabel('Generacja')
    plt.ylabel('Zysk Netto (Fitness)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()
    