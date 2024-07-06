import rustworkx as rx
import networkx as nx
import random
import math
import os

from qiskit import qasm2, transpile, QuantumCircuit, QuantumRegister
from qiskit.converters import dag_to_circuit, circuit_to_dag

from vfsexp import Vf 

def qubits_num(Circuit): #Circuit: gates list 
	num = max(max(gate) for gate in Circuit)
	num += 1
	return num

def CreateCircuitFromQASM(file, path):
    QASM_file = open(path + file, 'r')
    iter_f = iter(QASM_file)
    QASM = ''
    for line in iter_f: 
        QASM = QASM + line
    #print(QASM)
    cir = QuantumCircuit.from_qasm_str(QASM)
    QASM_file.close    
    return cir

def get_rx_subg_mapping(graph_max, G):
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
    while(True):
        try :
            item = next(vf2)
            reverse_mapping = {rx_nx_s[value]: rx_nx_G[key] for  key, value in item.items()}
            M.append(reverse_mapping)
        except StopIteration:
            break
    random_element = random.choice(M)
    return random_element

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
		merge_gates = sum(gate_layer_list[last_index:i+1], [])
		tmp_graph = nx.Graph()
		tmp_graph.add_edges_from(merge_gates)
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
    for cnot in gate_list:
        circ.cx(cnot[0], cnot[1])
    
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

def complete_mapping(cur_map, pre_map, post_map, indices, coupling_graph):
	unoccupied = [value for value in coupling_graph.nodes() if value not in cur_map]
	for index in indices:
		flag = False
		if pre_map:  #If pre_map is not empty
			if pre_map[index] in unoccupied:
				cur_map[index] = pre_map[index]
				flag = True
				unoccupied.remove(cur_map[index])
		if post_map and flag == False:
			if post_map[index] != -1 and post_map[index] in unoccupied:
				cur_map[index] = post_map[index]
				unoccupied.remove(cur_map[index])
				flag = True
		if flag == False:
			if pre_map:
				source = pre_map[index]
				node_of_shortest = dict()
				for node in unoccupied:
					distance = nx.shortest_path_length(coupling_graph, source=source, target=node)
					node_of_shortest[node] = distance
				min_node = min(node_of_shortest, key=node_of_shortest.get)
				cur_map[index] = min_node
				unoccupied.remove(min_node)
			elif post_map:
				source = post_map[index]
				node_of_shortest = dict()
				for node in unoccupied:
					distance = nx.shortest_path_length(coupling_graph, source=source, target=node)
					node_of_shortest[node] = distance
				min_node = min(node_of_shortest, key=node_of_shortest.get)
				cur_map[index] = min_node
				unoccupied.remove(min_node)
			else:
				min_node = random.choice(unoccupied)
				cur_map[index] = min_node
				unoccupied.remove(min_node)
