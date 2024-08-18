import rustworkx as rx
import networkx as nx
import numpy as np
import random
import math
import os
import re
import json

from qiskit import qasm2, transpile, QuantumCircuit, QuantumRegister
from qiskit.converters import dag_to_circuit, circuit_to_dag

from vfsexp import Vf 
import copy
from copy import deepcopy

def qubits_num(Circuit): #Circuit: gates list 
	num = max(max(gate) for gate in Circuit)
	num += 1
	return num

def CreateCircuitFromQASM(file, path):
	print(path+file)
	QASM_file = open(path + file, 'r')
	iter_f = iter(QASM_file)
	QASM = ''
	for line in iter_f: 
		QASM = QASM + line
	cir = QuantumCircuit.from_qasm_str(QASM)
	QASM_file.close    
	return cir

def get_rx_all_mapping(graph_max, G):
	sub_graph = rx.networkx_converter(graph_max)
	big_graph = rx.networkx_converter(G)
	nx_edge_s = list(graph_max.edges())
	rx_edge_s = list(sub_graph.edge_list())
	rx_nx_s = dict()
	for i in range(len(rx_edge_s)):
		if rx_edge_s[i][0] not in rx_nx_s:
			rx_nx_s[rx_edge_s[i][0]] = nx_edge_s[i][0]
		if rx_edge_s[i][1] not in rx_nx_s:
			rx_nx_s[rx_edge_s[i][1]] = nx_edge_s[i][1]
	nx_edge_G = list(G.edges())
	rx_edge_G = list(big_graph.edge_list())
	rx_nx_G = dict()
	for i in range(len(rx_edge_G)):
		if rx_edge_G[i][0] not in rx_nx_G:
			rx_nx_G[rx_edge_G[i][0]] = nx_edge_G[i][0]
		if rx_edge_G[i][1] not in rx_nx_G:
			rx_nx_G[rx_edge_G[i][1]] = nx_edge_G[i][1]
	vf2 = rx.vf2_mapping(big_graph, sub_graph, subgraph=True, induced = False)
	M = []
	times = 0
	while(True):
		try :
			times += 1
			if times % 100000 == 0:
				print(times)
			item = next(vf2)
			reverse_mapping = {rx_nx_s[value]: rx_nx_G[key] for  key, value in item.items()}
			M.append(reverse_mapping)
		except StopIteration:
			break
	return M

def get_rx_rand_mapping(graph_max, G):
	M = get_rx_all_mapping(graph_max, G)
	random_element = random.choice(M)
	return random_element

def get_rx_one_mapping(graph_max, G):
	sub_graph = rx.networkx_converter(graph_max)
	big_graph = rx.networkx_converter(G)
	nx_edge_s = list(graph_max.edges())
	rx_edge_s = list(sub_graph.edge_list())
	rx_nx_s = dict()
	for i in range(len(rx_edge_s)):
		if rx_edge_s[i][0] not in rx_nx_s:
			rx_nx_s[rx_edge_s[i][0]] = nx_edge_s[i][0]
		if rx_edge_s[i][1] not in rx_nx_s:
			rx_nx_s[rx_edge_s[i][1]] = nx_edge_s[i][1]
	nx_edge_G = list(G.edges())
	rx_edge_G = list(big_graph.edge_list())
	rx_nx_G = dict()
	for i in range(len(rx_edge_G)):
		if rx_edge_G[i][0] not in rx_nx_G:
			rx_nx_G[rx_edge_G[i][0]] = nx_edge_G[i][0]
		if rx_edge_G[i][1] not in rx_nx_G:
			rx_nx_G[rx_edge_G[i][1]] = nx_edge_G[i][1]
	vf2 = rx.vf2_mapping(big_graph, sub_graph, subgraph=True, induced = False)
	item = next(vf2)
	reverse_mapping = {rx_nx_s[value]: rx_nx_G[key] for  key, value in item.items()}
	return reverse_mapping

def map_dist(map1, map2, coupling_graph):
	dist = 0
	for i in range(len(map1)):
		if map1[i] != -1 and map2[i] != -1:
			dist += nx.shortest_path_length(coupling_graph, map1[i], map2[i])
	return dist

def get_close_mapping_rx(subG, G, pre_map, num_q):
	M = get_rx_all_mapping(subG, G)
	print("num of M", len(M))
	if not M:
		raise()
	min_dis = np.inf
	for mapping in M:
		dist = map_dist(pre_map, map2list(mapping,num_q), G)
		if dist < min_dis:
			final_map = map2list(mapping,num_q)
			min_dis = dist
	return final_map


def rx_is_subgraph_iso(G, subG):
    Grx = rx.networkx_converter(G)
    subGrx = rx.networkx_converter(subG)
    gm = rx.is_subgraph_isomorphic(Grx, subGrx, induced = False)   
    return gm

def get_layer_gates(dag):
    gate_layer_list = []
    for item in dag.layers():
        gate_layer = []
        graph_one = nx.Graph()
        for gate in item['partition']:
            c0 = gate[0]._index
            c1 = gate[1]._index
            gate_layer.append([c0, c1])
        gate_layer_list.append(gate_layer)
    return gate_layer_list

def parition_from_DAG(dag, coupling_graph):
	gate_layer_list = get_layer_gates(dag)
	num_of_gate = 0
	last_index = 0
	partition_gates = []
	for i in range(len(gate_layer_list)):
		#print(i)
		#print(last_index)
		merge_gates = sum(gate_layer_list[last_index:i+1], [])
		tmp_graph = nx.Graph()
		tmp_graph.add_edges_from(merge_gates)
		connected_components = list(nx.connected_components(tmp_graph))
		isIso = True
		for idx, component in enumerate(connected_components, 1):
			subgraph = tmp_graph.subgraph(component)
			if len(subgraph.edges()) == nx.diameter(subgraph): #path-tolopology, must sub_iso
				continue
			if not rx_is_subgraph_iso(coupling_graph, subgraph):
				isIso = False
				break
		if isIso:
			if i == len(gate_layer_list) - 1:
				merge_gates = sum(gate_layer_list[last_index: i+1], [])
				partition_gates.append(merge_gates)
			continue
		else:
			merge_gates = sum(gate_layer_list[last_index: i], [])
			partition_gates.append(merge_gates)
			last_index = i
			if i == len(gate_layer_list) - 1:
				merge_gates = sum(gate_layer_list[last_index: i+1], [])
				partition_gates.append(merge_gates)

	return partition_gates

def partition_from_ini(dag, coupling_graph, initial_mapping):
	gate_layer_list = get_layer_gates(dag)
	last_index = 0
	partition_gates = []
	first_part = []
	for i in range(len(gate_layer_list)):
		flag = True
		for gate in gate_layer_list[i]:
			if (initial_mapping[gate[0]],initial_mapping[gate[1]]) in coupling_graph.edges():
				continue
			else:
				flag = False
				break
		if flag:
			first_part.extend(gate_layer_list[i])
		else:
			last_index = i
			break
	partition_gates.append(first_part)
	begin_index = last_index
	for i in range(begin_index, len(gate_layer_list)):
		merge_gates = sum(gate_layer_list[last_index:i+1], [])
		tmp_graph = nx.Graph()
		tmp_graph.add_edges_from(merge_gates)
		connected_components = list(nx.connected_components(tmp_graph))
		isIso = True
		for idx, component in enumerate(connected_components, 1):
			subgraph = tmp_graph.subgraph(component)
			if len(subgraph.edges()) == nx.diameter(subgraph): #path-tolopology, must sub_iso
				continue
			if not rx_is_subgraph_iso(coupling_graph, subgraph):
				isIso = False
				break
		if isIso:
			if i == len(gate_layer_list) - 1:
				merge_gates = sum(gate_layer_list[last_index: i+1], [])
				partition_gates.append(merge_gates)
			continue
		else:
			merge_gates = sum(gate_layer_list[last_index: i], [])
			partition_gates.append(merge_gates)
			last_index = i
			if i == len(gate_layer_list) - 1:
				merge_gates = sum(gate_layer_list[last_index: i+1], [])
				partition_gates.append(merge_gates)

	return partition_gates


def get_max_diamter(G):
	diameter = []
	connected_components = list(nx.connected_components(G))
	# 对每个联通分支计算直径
	for idx, component in enumerate(connected_components, 1):
    	# 创建联通分支的子图
		subgraph = G.subgraph(component)
    
    	# 计算直径
		diameter.append(nx.diameter(subgraph))
	return max(diameter)

def partition_G_property(dag, coupling_graph):
	gate_layer_list = get_layer_gates(dag)
	num_of_gate = 0
	last_index = 0
	partition_gates = []
	for i in range(len(gate_layer_list)):
		merge_gates = sum(gate_layer_list[last_index:i+1], [])
		tmp_graph = nx.Graph()
		tmp_graph.add_edges_from(merge_gates)
		diameter = get_max_diamter(tmp_graph)
		if diameter <= 10 or (len(tmp_graph.edges())/diameter) > 2.8:
			if rx_is_subgraph_iso(coupling_graph, tmp_graph):
				if i == len(gate_layer_list) - 1:
					merge_gates = sum(gate_layer_list[last_index: i+1], [])
					partition_gates.append(merge_gates)
				continue
			else:
				merge_gates = sum(gate_layer_list[last_index: i], [])
				partition_gates.append(merge_gates)
				last_index = i
				if i == len(gate_layer_list) - 1:
					merge_gates = sum(gate_layer_list[last_index: i+1], [])
					partition_gates.append(merge_gates)

		else:
			merge_gates = sum(gate_layer_list[last_index: i], [])
			partition_gates.append(merge_gates)
			last_index = i
			if i == len(gate_layer_list) - 1:
				merge_gates = sum(gate_layer_list[last_index: i+1], [])
				partition_gates.append(merge_gates)
	return partition_gates


def get_2q_gates_list(circ):
	gate_2q_list = []
	instruction = circ.data
	for ins in instruction:
		if ins.operation.num_qubits == 2:
			gate_2q_list.append((ins.qubits[0]._index, ins.qubits[1]._index))
	return gate_2q_list


def get_qubits_num(gate_2q_list):  
    num = max(max(gate) for gate in gate_2q_list)
    num += 1
    return num

def gates_list_to_QC(gate_list):  #default all 2-q gates circuit
    Lqubit = get_qubits_num(gate_list)
    circ = QuantumCircuit(Lqubit)
    # issue: cz
    for two_qubit_gate in gate_list:
        circ.cz(two_qubit_gate[0], two_qubit_gate[1])
    
    dag = circuit_to_dag(circ)
    return circ, dag


def euclidean_distance(node1, node2):
	x1, y1 = node1
	x2, y2 = node2
	return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_grid_with_Rb(n, m, Rb):
    G = nx.grid_2d_graph(n, m)  # 生成n*m的网格图
    for node1 in G.nodes():
    	for node2 in G.nodes():
    		if node1 != node2:
    			distance = euclidean_distance(node1, node2)
    			if distance <= Rb:
    				G.add_edge(node1, node2)

    return G

def extend_graph(coupling_graph, arch_size, Rb):
	coupling_graph = generate_grid_with_Rb(arch_size+1, arch_size+1, Rb)
	return coupling_graph

def get_embedding(gates_list, previous_embedding, coupling_graph):
	tmp_graph = nx.Graph()
	tmp_graph.add_edges_from(gates_list)
	vf2 = Vf(tmp_graph, coupling_graph, {}, 50, preMap=previous_embedding, upperbound=1000)
	max_mapping = vf2.dfsMatchBest({}) #Times
	return max_mapping

def map2list(mapping, num_q):
	map_list = [-1] * num_q
	for key, value in mapping.items():
		map_list[key] = value

	return map_list

def complete_mapping(i, embeddings, indices, coupling_graph):
	cur_map = embeddings[i]
	unoccupied = [value for value in coupling_graph.nodes() if value not in cur_map]
	for index in indices:
		flag = False
		if i != 0:  #If pre_map is not empty
			if embeddings[i-1][index] in unoccupied:
				cur_map[index] = embeddings[i-1][index]
				flag = True
				unoccupied.remove(cur_map[index])
		if i != len(embeddings) - 1 and flag == False:
			for j in range(i+1, len(embeddings)):
				if embeddings[j][index] != -1 and embeddings[j][index] in unoccupied:
					cur_map[index] = embeddings[j][index]
					unoccupied.remove(cur_map[index])
					flag = True
					break
		if flag == False:
			if i != 0:
				source = embeddings[i-1][index]
				node_of_shortest = dict()
				for node in unoccupied:
					distance = nx.shortest_path_length(coupling_graph, source=source, target=node)
					node_of_shortest[node] = distance
				min_node = min(node_of_shortest, key=node_of_shortest.get)
				cur_map[index] = min_node
				unoccupied.remove(min_node)
				flag = True
			elif i != len(embeddings) - 1:
				for j in range(i+1, len(embeddings)):
					if embeddings[j][index] != -1:
						source = embeddings[j][index]
						node_of_shortest = dict()
						for node in unoccupied:
							distance = nx.shortest_path_length(coupling_graph, source=source, target=node)
							node_of_shortest[node] = distance
						min_node = min(node_of_shortest, key=node_of_shortest.get)
						cur_map[index] = min_node
						unoccupied.remove(min_node)
						flag = True
						break
		if flag == False:
			min_node = random.choice(unoccupied)
			cur_map[index] = min_node
			unoccupied.remove(min_node)
	return cur_map


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
    movement_result = []
    layer = copy.deepcopy(layer)
    for qubit in resolution_order:
        sorted_keys.remove(qubit)
        
        move = movements[qubit]
        movement_result.append([qubit, [move[0],move[1]],[move[2],move[3]]])
        # print(f'Move qubit {qubit} from ({move[0]}, {move[1]}) to ({move[2]}, {move[3]})')
        for qubit_ in layer["qubits"]:
            if qubit_["id"] == qubit:
                qubit_["a"] = 1
        
        # Remove resolved violations
        violations = [v for v in violations if qubit not in v]
        del movements[qubit]
    
    return layer, movements, violations, movement_result

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


def loc_to_qasm(n: int, qubit: int, loc: tuple[int, int]) -> str:
    """
    Converts a qubit location to a QASM formatted string.

    Parameters:
    n (int): The number of qubits in the quantum register.
    qubit (int): The specific qubit index.
    loc (tuple[int, int]): The location of the qubit as a tuple of two integers.

    Returns:
    str: The QASM formatted string representing the qubit location.

    Raises:
    ValueError: If the loc tuple does not have exactly two elements.
    """
    if len(loc) != 2:
        raise ValueError("Invalid loc, it must be a tuple of length 2")
    return f"Qubit(QuantumRegister({n}, 'q'), {qubit})\n({loc[0]}, {loc[1]})"

def map_to_qasm(n: int, map: list[tuple[int, int]], filename: str) -> None:
    """
    Converts a list of qubit locations to QASM format and saves it to a file.

    Parameters:
    n (int): The number of qubits in the quantum register.
    map (list[tuple[int, int]]): A list of tuples representing the locations of the qubits.
    filename (str): The name of the file to save the QASM formatted strings.

    Returns:
    None
    """
    with open(filename, 'w') as f:
        for i in range(n):
            f.write(loc_to_qasm(n, i, map[i]) + '\n')
def gate_in_layer(gate_list:list[list[int]])->list[map]:
    res = []
    for i in range(len(gate_list),-1):
        assert len(gate_list[i]) == 2
        res.append({'id':i,'q0':gate_list[i][0],'q1':gate_list[i][1]})
    return res
            
def check_available(graph, coupling_graph, mapping):

	for eg0, eg1 in graph.edges():
		if (mapping[eg0], mapping[eg1]) not in coupling_graph.edges():
			return False
	return True

def check_intersect(gate1, gate2, coupling_graph, mapping):
	rg1 = 1/2 * (euclidean_distance(mapping[gate1[0]], mapping[gate1[1]]))
	rg2 = 1/2 * (euclidean_distance(mapping[gate2[0]], mapping[gate2[1]]))
	dis = rg1 + rg2
	if euclidean_distance(mapping[gate1[0]], mapping[gate2[0]]) >= dis and \
		euclidean_distance(mapping[gate1[0]], mapping[gate2[1]]) >= dis and \
		euclidean_distance(mapping[gate1[1]], mapping[gate2[0]]) >= dis and \
		euclidean_distance(mapping[gate1[1]], mapping[gate2[1]]) >= dis:
		return True
	else:
		return False

def check_intersect_ver2(gate1, gate2, coupling_graph, mapping, r_re):
	if euclidean_distance(mapping[gate1[0]], mapping[gate2[0]]) > r_re and \
		euclidean_distance(mapping[gate1[0]], mapping[gate2[1]]) > r_re and \
		euclidean_distance(mapping[gate1[1]], mapping[gate2[0]]) > r_re and \
		euclidean_distance(mapping[gate1[1]], mapping[gate2[1]]) > r_re:
		return True
	else:
		return False

def get_parallel_gates(gates, coupling_graph, mapping, r_re):
	gates_list = []
	_, dag = gates_list_to_QC(gates)
	gate_layer_list = get_layer_gates(dag)

	for items in gate_layer_list:
		gates_copy = deepcopy(items)
		while(len(gates_copy) != 0):
			parallel_gates = []
			parallel_gates.append(gates_copy[0])
			for i in range(1, len(gates_copy)):
				flag = True
				for gate in parallel_gates:
					if check_intersect_ver2(gate, gates_copy[i], coupling_graph, mapping, r_re):
						continue
					else:
						flag = False
						break
				if flag:
					parallel_gates.append(gates_copy[i])

			for gate in parallel_gates:
				gates_copy.remove(gate)
			gates_list.append(parallel_gates)
			#print("parl:",parallel_gates)
	return gates_list

def set_parameters(default):
	para = {}
	if default:
		para['T_cz'] = 0.2  #us
		para['T_eff'] = 1.5e6 #us
		para['T_trans'] = 20 # us
		para['AOD_width'] = 3 #um
		para['AOD_height'] = 3 #um
		para['Move_speed'] = 0.55 #um/us
		para['F_cz'] = 0.995 

	return para

def compute_fidelity(parallel_gates, all_movements, num_q, gate_num):
	para = set_parameters(True)
	t_total = 0
	t_total += (len(parallel_gates) * para['T_cz']) # cz execution time, parallel
	t_move = 0
	for move in all_movements:
		t_total += (4 * para['T_trans']) # pick/drop/pick/drop
		t_move += (4 * para['T_trans'])
		max_dis = 0
		for each_move in move:
			x1, y1 = each_move[1][0],each_move[1][1]
			x2, y2 = each_move[2][0],each_move[2][1]
			dis = (abs(x2-x1)*para['AOD_width'])**2 + (abs(y2-y1)*para['AOD_height'])**2
			if dis > max_dis:
				max_dis = dis
		max_dis = math.sqrt(max_dis)
		t_total += (max_dis/para['Move_speed'])
		t_move += (max_dis/para['Move_speed'])

	t_idle = num_q * t_total - gate_num * para['T_cz']
	Fidelity = math.exp(-t_idle/para['T_eff']) * (para['F_cz']**gate_num)
	move_fidelity = math.exp(-t_move/para['T_eff'])
	return t_idle, Fidelity, move_fidelity

def get_embeddings(partition_gates, coupling_graph, num_q, arch_size, Rb, initial_mapping=None):
	embeddings = []
	begin_index = 0
	extend_position = []
	if initial_mapping:
		embeddings.append(initial_mapping)
		begin_index = 1
	for i in range(begin_index, len(partition_gates)):
		tmp_graph = nx.Graph()
		tmp_graph.add_edges_from(partition_gates[i])
		if not rx_is_subgraph_iso(coupling_graph, tmp_graph):
			coupling_graph = extend_graph(coupling_graph, arch_size, Rb)
			extend_position.append(i)
		next_embedding = get_rx_one_mapping(tmp_graph, coupling_graph)
		next_embedding = map2list(next_embedding,num_q)
		embeddings.append(next_embedding)

	for i in range(begin_index, len(embeddings)):
		indices = [index for index, value in enumerate(embeddings[i]) if value == -1]
		if indices:
			embeddings[i] = complete_mapping(i, embeddings, indices, coupling_graph)

	return embeddings, extend_position

def qasm_to_map(filename):

	with open(filename, 'r') as file:
		lines = file.readlines()
	
    # 确保行数为偶数
	if len(lines) % 2 != 0:
		raise ValueError("文件内容的行数应为偶数，以便正确配对比特和映射位置。")
	qubit_pattern = re.compile(r"Qubit\(QuantumRegister\((\d+), 'q'\), (\d+)\)")
	match = qubit_pattern.search(lines[0].strip())
	num_q = int(match.group(1))
	mapping = [-1]*num_q
    # 遍历每对行
	for i in range(0, len(lines), 2):
        # 读取比特行和映射位置行
		qubit_line = lines[i].strip()
		position_line = lines[i+1].strip()
        
        # 解析比特索引
		match = qubit_pattern.search(qubit_line)
		if match:
			num_q = int(match.group(1))
			bit_index = int(match.group(2))
		else:
			raise ValueError(f"无法从比特行提取索引: {qubit_line}")
        # 解析映射位置
		try:
			position = eval(position_line)
		except Exception as e:
			raise ValueError(f"解析映射位置时出错: {position_line}") from e
        
        # 扩展列表到足够的长度
		mapping[bit_index] = position
    
	return mapping

def compute_fidelity_tetris(cycle_file, qasm_file, path):

	with open(path+cycle_file, 'r') as cyc_file:
		cyc_lines = cyc_file.readlines()

	last_line = cyc_lines[-1].strip()
	last_gate = list(map(int, last_line.split()))

	cnot_count = 0
	swap_count = 0
	circ = CreateCircuitFromQASM(qasm_file, path)
	for instruction, qargs, cargs in circ.data:
		if instruction.name == 'cx':  # CNOT 门
			cnot_count += 1
		elif instruction.name == 'swap':  # SWAP 门
			swap_count += 1

	num_q = len(circ.qubits)
	gate_num = cnot_count + 3*swap_count
	
	para = set_parameters(True)
	gate_cycle = (last_gate[0]+1)/2
	t_total = gate_cycle*para['T_cz']
	t_idle = num_q * t_total - gate_num * para['T_cz']

	Fidelity =math.exp(-t_idle/para['T_eff']) * (para['F_cz']**gate_num)
	print(Fidelity)
	return Fidelity, swap_count, gate_cycle

def write_data(data, path, file_name): 
	with open(path+file_name, 'w') as file:
		for sublist in data:
        # 将每个子列表转换为 JSON 格式的字符串，并写入文件
			file.write(json.dumps(sublist) + '\n')

def read_data(path, file_name):
	with open(path+file_name, 'r') as file:
    # 逐行读取文件
		data = [json.loads(line) for line in file]

# 输出读取的数据
	return data

def get_circuit_from_json(num_qubits: int) -> QuantumCircuit:
    """
    Load a quantum circuit from a JSON file based on the number of qubits.

    Args:
        num_qubits (int): The number of qubits for the desired circuit.

    Returns:
        QuantumCircuit: The loaded quantum circuit.

    Raises:
        ValueError: If no circuit configuration is found for the specified number of qubits.
    """
    # Path to the JSON file containing the circuits
    json_file_path = "./Enola/graphs.json"
    with open(json_file_path, "r") as file:
        data = json.load(file)
    circuit_data = data.get(str(num_qubits))

    # Check if the circuit configuration exists
    if circuit_data:
        quantum_circuit = QuantumCircuit(num_qubits)
        # Add gates to the circuit based on the data
        for layer in circuit_data:
            for gate in layer:
                quantum_circuit.cz(gate[0], gate[1])
        return quantum_circuit
    else:
        # Raise an error if no configuration is found
        raise ValueError(f"No circuit configuration for {num_qubits} qubits in graphs.json")