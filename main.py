from naqst import *

if __name__ == "__main__":

	#qasm input
	path = "Data/RevLib/"
	files = os.listdir(path)
	file_name = files[1]
	print(file_name)
	circuit = CreateCircuitFromQASM(file_name, path)
	#transform to cz-based circuit
	cz_circuit = transpile(circuit, basis_gates=['cz', 'rx', 'ry', 'rz', 'h', 't'])
	#cz gates list
	gate_2q_list = get_2q_gates_list(cz_circuit)
	#print(gate_2q_list)
	#obtain corresponding DAG
	_, dag = gates_list_to_QC(gate_2q_list)
	#gate_num = len(gate_2q_list)
	#obtain the qubits number
	num_q = qubits_num(gate_2q_list)
	#print("Num of gates", gate_num)

	arch_size = 3
	Rb = math.sqrt(2)
	#obtain the corresponding coupling_graph 
	coupling_graph = generate_grid_with_Rb(arch_size,arch_size, Rb)

	#obtain the gates partition
	partition_gates = parition_from_DAG(dag, coupling_graph)
	print("------------------------------")
	part_gate_num = 0
	for gates in partition_gates:
		part_gate_num += len(gates)
		#print(gates)
	print("final num is:", part_gate_num)

    #for each partition, find a proper embedding
	embeddings = []

	for i in range(len(partition_gates)):
		tmp_graph = nx.Graph()
		tmp_graph.add_edges_from(partition_gates[i])
		next_embedding = get_rx_subg_mapping(tmp_graph, coupling_graph)
		embeddings.append(map2list(next_embedding,num_q))
	print(embeddings)

	for i in range(len(embeddings)):
		indices = [index for index, value in enumerate(embeddings[i]) if value == -1]
		if indices:
			if i != 0:
				pre_map = embeddings[i-1]
			else:
				pre_map = []
			if i != len(embeddings) -1:
				post_map = embeddings[i+1]
			else:
				post_map = []
			embeddings[i] = complete_mapping(i, embeddings, indices, coupling_graph)
			
	print(embeddings)
	# TODO: for the subsequent embeddings, use AOD to move from current embedding to next embedding
    
	window = False
	window_size = 1000
	routing_strategy = "maximalis"
	layers = []
	layers.append(map_to_layer(embeddings[0]))
	for i in range(len(embeddings) - 1):
		current_map = embeddings[i]
		next_map = embeddings[i + 1]
		next_layer = map_to_layer(next_map)
		
		last_layer = copy.deepcopy(layers[-1])
		movements = get_movement(current_map,next_map)
		# Sort movements by distance in descending order
		sorted_keys = sorted(movements.keys(), key=lambda k: math.dist((movements[k][0], movements[k][1]), (movements[k][2], movements[k][3])), reverse=False)
		# print(f'sorted_keys:{sorted_keys}')
		
		# Check for violations
		violations = []
		for i in range(len(sorted_keys)):
			for j in range(i + 1, len(sorted_keys)):
				if not compatible_2D(movements[sorted_keys[i]], movements[sorted_keys[j]]):
					violations.append((sorted_keys[i], sorted_keys[j]))

		# print(f'Violations: {violations}')
		
		# Resolve violations
		while violations:
			new_layer,movements,violations = solve_violations(movements,violations,sorted_keys,routing_strategy,num_q,last_layer)
			layers.append(new_layer)
			for i in range(num_q):
				if last_layer["qubits"][i]["a"] == 1:
					last_layer["qubits"][i] = next_layer
				
		if movements:
			for qubit in movements:
				move = movements[qubit]
				for qubit_ in last_layer["qubits"]:
					if qubit_["id"] == qubit:
						qubit_["a"] = 1
			layers.append(last_layer)
		layers.append(next_layer)
		
				
	data = {
		# "runtime": float(time.time() - start_time),
		"no_transfer": False,
		"layers": layers,
		"n_q": num_q,
		# "g_q": list_gates,
	}
	print(data)




