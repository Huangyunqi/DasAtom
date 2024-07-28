from naqst import *
from openpyxl import Workbook
import math
import time
from vfsexp import Vf
from Enola.codegen import CodeGen, global_dict
import json
from SA import find_map_SA

if __name__ == "__main__":

	#qasm input
	#path = "Data/qft/"
	path = "Data/Tetris_cz/"
	# path = "Data/Q_tetris/"
	files = os.listdir(path)
	#file_name = 'qft_50.qasm'
	for num_file in range(len(files)):
		print(f"begin:{num_file} of {len(files)}")
		#file_name = 'qft_{}.qasm'.format(num_file)
		file_name = files[num_file]
		print(file_name)
		#wb = Workbook()
		#ws = wb.active
		total_time = time.time()
		circuit = CreateCircuitFromQASM(file_name, path)
	#transform to cz-based circuit
		# cz_circuit = transpile(circuit, basis_gates=['cz', 'rx', 'ry', 'rz', 'h', 't'])
	#cz gates list
		# print(circuit.draw())
		gate_2q_list = get_2q_gates_list(circuit)
	#print(gate_2q_list)
		gate_num = len(gate_2q_list)
		print("Num of gates", gate_num)
	#obtain corresponding DAG
		_, dag = gates_list_to_QC(gate_2q_list)
	#obtain the qubits number
		num_q = qubits_num(gate_2q_list)

		arch_size = math.ceil(math.sqrt(num_q))
		Rb = math.sqrt(2)
	#obtain the corresponding coupling_graph 
		coupling_graph = generate_grid_with_Rb(arch_size,arch_size, Rb)

	#obtain the gates partition
		time_part = time.time()
		partition_gates = parition_from_DAG(dag, coupling_graph)
		time_part1 = time.time()
		print("partition time is, ",time_part1-time_part)
		#ws.append(["partition time", time_part1-time_part])
	#print("------------------------------")
		part_gate_num = 0
		for gates in partition_gates:
	#		print("------")
			part_gate_num += len(gates)
	#	print(gates)
		print("final num is:", part_gate_num)


    #for each partition, find a proper embedding
		embeddings = []
		print("partition number:", len(partition_gates))
		for i in range(0, len(partition_gates)):
			data = []
			tmp_graph = nx.Graph()
			tmp_graph.add_edges_from(partition_gates[i])
			# print(tmp_graph.edges())
			data.append(str(partition_gates[i]))
			time1 = time.time()
			next_embedding = get_rx_one_mapping(tmp_graph, coupling_graph)
			next_embedding = map2list(next_embedding,num_q)
			#next_embedding = find_map_SA(embeddings[i-1], tmp_graph, coupling_graph)
			#print("SA embedding:",next_embedding)
			#if not check_available(tmp_graph, coupling_graph, next_embedding):
			#	print("SA False!!!!")
			#	ws.append(["!SA False!"])
			#	next_embedding = get_rx_one_mapping(tmp_graph, coupling_graph)
			#	next_embedding = map2list(next_embedding,num_q)
			time2 = time.time()
			embeddings.append(next_embedding)
		#embeddings.append(next_embedding)
			data.append(str(embeddings[-1]))
			data.append(time2-time1)
			#ws.append(data)
			# print(embeddings[-1])
			# print(time2-time1)
		
		for i in range(len(embeddings)):
			indices = [index for index, value in enumerate(embeddings[i]) if value == -1]
			data = []
			time1 = time.time()
			if indices:
				embeddings[i] = complete_mapping(i, embeddings, indices, coupling_graph)
			time2 = time.time()
			data.append(str(embeddings[i]))
			data.append(time2-time1)
			#ws.append(data)
		total_time1 = time.time()
		print("total time is:", total_time1-total_time)
		#ws.append(["total time", total_time1-total_time])
		#save_file = 'results/yq_test/qft/qft_{}_SA.xlsx'.format(num_file)
		#print(save_file)
		#wb.save(save_file)
		map_to_qasm(num_q,embeddings[0],f"results/initial_map/{files[num_file]}")
  # TODO: for the subsequent embeddings, use AOD to move from current embedding to next embedding
    
		global_dict['full_code'] = True

		window = False
		window_size = 1000
		routing_strategy = "maximalis"
		import copy

		layers = []
		initial_map = map_to_layer(embeddings[0])
		initial_map["gates"] = gate_in_layer(partition_gates[0])
		# print(initial_map)
		layers.append(initial_map)
		data = {
				"no_transfer": False,
				"layers": layers,
				"n_q": num_q,
				"g_q": gate_2q_list,
		}
		data['n_x'] = arch_size
		data['n_y'] = arch_size
		data['n_r'] = arch_size
		data['n_c'] = arch_size
		# print(layers)
		codegen = CodeGen(data)
		program = codegen.builder(no_transfer=False)
		program = program.emit_full()
		for i in range(len(embeddings) - 1):
			layers = []
			current_map = embeddings[i]
			next_map = embeddings[i + 1]
			last_layer = map_to_layer(current_map)
			next_layer = map_to_layer(next_map)
			movements = get_movement(current_map,next_map)
			# Sort movements by distance in descending order
			sorted_keys = sorted(movements.keys(), key=lambda k: math.dist((movements[k][0], movements[k][1]), (movements[k][2], movements[k][3])), reverse=False)
			# print(f'sorted_keys:{sorted_keys}')
			# Check for violations
			violations = []
			for i_key in range(len(sorted_keys)):
				for j_key in range(i_key + 1, len(sorted_keys)):
					if not compatible_2D(movements[sorted_keys[i_key]], movements[sorted_keys[j_key]]):
						violations.append((sorted_keys[i_key], sorted_keys[j_key]))

			# print(f'Violations: {violations}')
			# Resolve violations
			while violations:
				new_layer,movements,violations = solve_violations(movements,violations,sorted_keys,routing_strategy,num_q,last_layer)
				layers.append(new_layer)
				for q in range(num_q):
					if new_layer["qubits"][q]["a"] == 1:
						last_layer["qubits"][q] = next_layer["qubits"][q]
			# print(f"layers:{layers}")
			# print(f"last later:{last_layer}")
			if movements:
				for qubit in movements:
					move = movements[qubit]
					for qubit_ in last_layer["qubits"]:
						if qubit_["id"] == qubit:
							qubit_["a"] = 1
				layers.append(last_layer)
			layers[len(layers)-1]["gates"] = gate_in_layer(partition_gates[i+1])
			# layers.append(next_layer)

			# print(layers)
			data = {
				"no_transfer": False,
				"layers": layers,
				"n_q": num_q,
				"g_q": gate_2q_list,
			}
			data['n_x'] = arch_size
			data['n_y'] = arch_size
			data['n_r'] = arch_size
			data['n_c'] = arch_size
			codegen = CodeGen(data)
			temp = codegen.builder(no_transfer=False)
			temp = temp.emit_full()
			program += temp[2:]
		if global_dict["full_code"]:
			with open(f"results/full_code/{files[num_file]}_{num_q}_{0}_code_full.json", 'w') as f:
				json.dump(program, f)
				for instruction in program:
					instruction["state"] = {}
    # optional
    # run following command in terminal:
    # python Enola/animation.py f"Data/test_{num_q}_{0}_code_full.json" --dir "./Data/"



