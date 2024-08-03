import json
import math
import copy
from codegen import CodeGen, global_dict
from networkx import maximal_independent_set, Graph

def compatible_2D(a: list[int], b: list[int]) -> bool:
    """
    Checks if two 2D points are compatible based on specified rules.

    Parameters:
    a (list[int]): A list of four integers representing the first point. The elements are ordered as [x_loc_before, y_loc_before, x_loc_after, y_loc_after].
    b (list[int]): A list of four integers representing the second point. The elements are ordered as [x_loc_before, y_loc_before, x_loc_after, y_loc_after].

    Returns:
    bool: True if the points are compatible, False otherwise.
    """
    assert len(a) == 4 and len(b) == 4, "Both arguments must be lists with exactly four elements."

    # Check compatibility for the first two elements of each point
    if a[0] == b[0] and a[2] != b[2]:
        return False
    if a[2] == b[2] and a[0] != b[0]:
        return False
    if a[0] < b[0] and a[2] >= b[2]:
        return False
    if a[0] > b[0] and a[2] <= b[2]:
        return False

    # Check compatibility for the last two elements of each point
    if a[1] == b[1] and a[3] != b[3]:
        return False
    if a[3] == b[3] and a[1] != b[1]:
        return False
    if a[1] < b[1] and a[3] >= b[3]:
        return False
    if a[1] > b[1] and a[3] <= b[3]:
        return False

    return True

def maximalis_solve_sort(n: int, edges: list[tuple[int]], nodes: set[int]) -> list[int]:
    """
    Finds a maximal independent set from the given graph nodes using a sorted approach.

    Parameters:
    n (int): Number of nodes in the graph. The nodes were expressed by integers from 0 to n-1.
    edges (list[tuple[int]]): List of edges in the graph, where each edge is a tuple of two nodes.
    nodes (set[int]): Set of nodes to consider for the maximal independent set.

    Returns:
    list[int]: List of nodes in the maximal independent set.
    """
    # Initialize conflict status for each node
    is_node_conflict = [False for _ in range(n)]
    
    # Create a dictionary to store neighbors of each node
    node_neighbors = {i: [] for i in range(n)}
    
    # Populate the neighbors dictionary
    for edge in edges:
        node_neighbors[edge[0]].append(edge[1])
        node_neighbors[edge[1]].append(edge[0])
    
    result = []
    for i in nodes:
        if is_node_conflict[i]:
            continue
        else:
            result.append(i)
            for j in node_neighbors[i]:
                is_node_conflict[j] = True
    return result

def maximalis_solve(nodes:list[int], edges:list[tuple[int]])-> list[int]:
    """
    Wrapper function to find a maximal independent set using the Graph class.

    Parameters:
    n (int): Number of nodes in the graph. The nodes were expressed by integers from 0 to n-1.
    edges (list[tuple[int]]): List of edges in the graph.

    Returns:
    list[int]: List of nodes in the maximal independent set.
    """
    G = Graph()
    for i in nodes:
        G.add_node(i)
    for edge in edges:
        G.add_edge(edge[0], edge[1])
    
    # Use a library function to find the maximal independent set
    result = maximal_independent_set(G, seed=0) 
    return result

def get_movement(current_map: list, next_map: list, window_size=None) -> map:
    """
    Determines the movements of qubits between two maps.

    Parameters:
    current_map (list): List of current positions of qubits.
    next_map (list): List of next positions of qubits.
    window_size (optional): Size of the window for movement calculations.

    Returns:
    map: A dictionary with qubit movements.
    """
    movements = {}
    # Determine movements of qubits
    for qubit, current_position in enumerate(current_map):
        next_position = next_map[qubit]
        if current_position != next_position:
            move_details = current_position + next_position
            movements[qubit] = move_details
    return movements

def solve_violations(movements, violations, sorted_keys, routing_strategy, num_q, layer):
    """
    Resolves violations in qubit movements based on the routing strategy.

    Parameters:
    movements (dict): Dictionary of qubit movements.
    violations (list): List of violations to be resolved.
    sorted_keys (list): List of qubit keys sorted based on priority.
    routing_strategy (str): Strategy to use for routing ('maximalis' or 'maximalis_sort').
    num_q (int): Number of qubits.
    layer (dict): Dictionary representing the current layer configuration.

    Returns:
    tuple: Updated layer, remaining movements, and unresolved violations.
    """
    if routing_strategy == "maximalis":
        resolution_order = maximalis_solve(sorted_keys, violations)
    else:
        resolution_order = maximalis_solve_sort(num_q, violations, sorted_keys)
    
    # print(f'Resolution Order: {resolution_order}')
    
    layer = copy.deepcopy(layer)
    for qubit in resolution_order:
        sorted_keys.remove(qubit)
        
        # move = movements[qubit]
        # print(f'Move qubit {qubit} from ({move[0]}, {move[1]}) to ({move[2]}, {move[3]})')
        for qubit_ in layer["qubits"]:
            if qubit_["id"] == qubit:
                qubit_["a"] = 1
        
        # Remove resolved violations
        violations = [v for v in violations if qubit not in v]
        del movements[qubit]
    
    return layer, movements, violations

def map_to_layer(map: list) -> map:
    """
    Converts a list of qubit positions to a layer dictionary.

    Parameters:
    map (list): List of qubit positions.

    Returns:
    map: Dictionary representing the layer configuration.
    """
    return {
        "qubits": [{
            "id": i,
            "a": 0,
            "x": map[i][0],
            "y": map[i][1],
            "c": map[i][0],
            "r": map[i][1],
        } for i in range(len(map))],
        "gates": []
    }

def gate_in_layer(gate_list:list[list[int]])->list[map]:
    res = []
    for i in range(len(gate_list)-1,-1,-1):
        assert len(gate_list[i]) == 2
        res.append({'id':i,'q0':gate_list[i][0],'q1':gate_list[i][1]})
    return res

class route:
    def __init__(self, num_q: int , embeddings: list[list[int]] ,gate_2q_list: list[list[int]], arch_size: list[int], routing_strategy="maximalis")->None:
        self.num_q = num_q
        # Iterate over each embedding in the list of embeddings
        for embed in embeddings:
            # Ensure each embedding contains locations for all qubits
            assert len(embed) == num_q, f"Each embedding must contain locations for all {num_q} qubits."

            # Check each location within the embedding
            for loc in embed:
                # Ensure each location is a list of two elements (x and y coordinates)
                assert len(loc) == 2, "Each location must be a list containing exactly two coordinates: [x, y]."
        self.embeddings = embeddings
        # Ensure the number of embeddings matches the number of two-qubit gates
        assert len(embeddings) == len(gate_2q_list), (
            "The number of embeddings should match the number of two-qubit gates in gate_2q_list."
        )
        self.gate_2q_list = gate_2q_list
        
        # Ensure the architecture size list contains exactly two elements for dimensions x and y
        assert len(arch_size) == 2, "Architecture size should be specified as a list with two elements: [x, y]."

        # Ensure the total number of locations in the architecture is sufficient to accommodate all qubits
        assert arch_size[0] * arch_size[1] >= num_q, (
            f"The product of the architecture dimensions x and y must be at least {num_q} to accommodate all qubits; "
            f"currently, it is {arch_size[0] * arch_size[1]}."
        )
        self.arch_size = arch_size
        
        self.routing_strategy = routing_strategy

    def initialize_data(self):
        initial_layer = map_to_layer(self.embeddings[0])
        initial_layer["gates"] = gate_in_layer(self.partition_gates[0])
        self.layers.append(initial_layer)
        program = self.generate_program([initial_layer])
        self.program = []
        self.program += program

    def generate_program(self, layers:list):
        data = {
            "no_transfer": False,
            "layers": layers,
            "n_q": self.num_q,
            "g_q": self.gate_2q_list,
        }
        data['n_x'] = self.arch_size[0]
        data['n_y'] = self.arch_size[1]
        data['n_r'] = self.arch_size[0]
        data['n_c'] = self.arch_size[1]
        codeGen = CodeGen(data)
        program = codeGen.builder(no_transfer=False)
        return program.emit_full()

    def process_embeddings(self):
        for i in range(len(self.embeddings) - 1):
            program = self.resolve_movements(i)
            self.program += program

    def resolve_movements(self, current_pos):
        next_pos = current_pos + 1
        movements = get_movement(self.embeddings[current_pos],self.embeddings[current_pos+1])
        sorted_keys = sorted(movements.keys(), key=lambda k: math.dist(movements[k][:2], movements[k][2:]))
        violations = self.check_violations(sorted_keys, movements)
        
        layers = self.handle_violations(violations, movements, sorted_keys, current_pos)
        layers[len(layers)-1]["gates"] = gate_in_layer(self.gate_2q_list[next_pos])
        return self.generate_program(layers)[:2]

    def handle_violations(self,violations,movements,sorted_keys,current_pos):
        current_layer = map_to_layer(self.embeddings[current_pos])
        next_layer = map_to_layer(self.embeddings[current_pos+1])
        layers = []
        while violations:
            new_layer,movements,violations = solve_violations(movements,violations,sorted_keys,self.routing_strategy,self.num_q,current_layer)
            layers.append(new_layer)
            for q in range(self.num_q):
                if new_layer["qubits"][q]["a"] == 1:
                    current_layer["qubits"][q] = next_layer["qubits"][q]
        print(f"layers:{layers}")
        print(f"last later:{current_layer}")
        if movements:
            for move_qubit in movements:
                for qubit in current_layer["qubits"]:
                    if qubit["id"] == move_qubit:
                        qubit["a"] = 1
            layers.append(current_layer)
        return layers
    
    def check_violations(self, sorted_keys, movements):
        violations = []
        for i_key in range(len(sorted_keys)):
            for j_key in range(i_key + 1, len(sorted_keys)):
                if not compatible_2D(movements[sorted_keys[i_key]], movements[sorted_keys[j_key]]):
                    violations.append((sorted_keys[i_key], sorted_keys[j_key]))
        return violations

    def save_program(self, filename):
        with open(filename, 'w') as file:
            json.dump(self.program, file)
            
    def run(self, filename: str)-> None:
        self.initialize_data()
        self.process_embeddings()
        self.save_program(filename)
    # def animation(self):
        