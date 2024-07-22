from naqst import *
from openpyxl import Workbook
import math
import time
from vfsexp import Vf
from SA import find_map_SA
from codegen import CodeGen

if __name__ == "__main__":

	#qasm input
	path = "Data/Q_Tetris/"
	#path = "bench/"
	files = os.listdir(path)
	#file_name = 'qft_50.qasm'
	for num_file in [0]:
	#for num_file in range(len(files)):
		#file_name = 'qft_{}.qasm'.format(num_file)
		file_name = files[num_file]
		print(file_name)
		wb = Workbook()
		ws = wb.active
		total_time = time.time()
		circuit = CreateCircuitFromQASM(file_name, path)
	#transform to cz-based circuit
		cz_circuit = transpile(circuit, basis_gates=['cz', 'rx', 'ry', 'rz', 'h', 't'])
	#cz gates list
		gate_2q_list = get_2q_gates_list(cz_circuit)
	#print(gate_2q_list)
	#obtain corresponding DAG
		_, dag = gates_list_to_QC(gate_2q_list)
		gate_num = len(gate_2q_list)
	#obtain the qubits number
		num_q = qubits_num(gate_2q_list)
		print("Num of gates", gate_num)

		arch_size = math.ceil(math.sqrt(num_q))
		#Rb = math.sqrt(2)
		Rb = 2
	#obtain the corresponding coupling_graph 
		coupling_graph = generate_grid_with_Rb(arch_size,arch_size, Rb)

	#obtain the gates partition
		time_part = time.time()
		partition_gates = parition_from_DAG(dag, coupling_graph)
		time_part1 = time.time()
		print("partition time is, ",time_part1-time_part)
		ws.append(["partition time", time_part1-time_part])
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
		for i in range(len(partition_gates)):
			data = []
			tmp_graph = nx.Graph()
			tmp_graph.add_edges_from(partition_gates[i])
			#print(tmp_graph.edges())
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
			ws.append(data)
			#print(embeddings[-1])
			#print(time2-time1)
		
		for i in range(len(embeddings)):
			indices = [index for index, value in enumerate(embeddings[i]) if value == -1]
			data = []
			time1 = time.time()
			if indices:
				embeddings[i] = complete_mapping(i, embeddings, indices, coupling_graph)
			time2 = time.time()
			#print(embeddings[i])
			data.append(str(embeddings[i]))
			#data.append(time2-time1)
			ws.append(data)
		total_time1 = time.time()
		print("total time is:", total_time1-total_time)
		ws.append(["total partition time", total_time1-total_time])
		#save_file = 'results/yq_test/qft/qft_{}_SA.xlsx'.format(num_file)
		#print(save_file)
		#wb.save(save_file)
    
  # TODO: for the subsequent embeddings, use AOD to move from current embedding to next embedding
    
		window = False
		window_size = 1000
		routing_strategy = "maximalis"
		layers = []
		layers.append(map_to_layer(embeddings[0]))
		atom_trans_num = 0
		total_duration = 0
		movement_results = []
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
				new_layer,movements,violations, movement_result = solve_violations(movements,violations,sorted_keys,routing_strategy,num_q,last_layer)
				movement_results.extend(movement_result)
				layers.append(new_layer)
				for i in range(num_q):
					if new_layer["qubits"][i]["a"] == 1:
						last_layer["qubits"][i] = next_layer["qubits"][i]
				
			if movements:
				for qubit in movements:
					move = movements[qubit]
					#print(f'Move qubit {qubit} from ({move[0]}, {move[1]}) to ({move[2]}, {move[3]})')
					movement_results.append([[qubit], [move[0],move[1]], [move[2],move[3]]])
					for qubit_ in last_layer["qubits"]:
						if qubit_["id"] == qubit:
							qubit_["a"] = 1
				layers.append(last_layer)
			layers.append(next_layer)
		#print(movement_results)
		for move_res in movement_results:
			print(move_res)
			atom_trans_num += 3 #include 1 active, 1 big_move, 1 deactive
			atom_trans_num += 3 # when need to exchange two qubits, need to put it around until the ori qubit out, then put into it
			total_duration += (6*15) #each atom trans remain 15us
			max_dis = 0
			data = []
			data.append(str(move_res))
			width_dis = abs(move_res[1][0]-move_res[2][0]) * 19
			heigh_dis = abs(move_res[1][1] - move_res[2][1]) * 15
			max_dis = max(width_dis, heigh_dis)
			ws.append(data)
			total_duration += 200 * ((max_dis)/110)**(1/2)
		Fidelity = 0.999**atom_trans_num * ((1-total_duration/(1.5e6))**num_q)
		ws.append(["Fidelity:", Fidelity])
		print("Fidelity is", Fidelity)
		save_file = 'results/yq_test/Tetris/{}_rb2.xlsx'.format(file_name)
		wb.save(save_file)

		data = {
		# "runtime": float(time.time() - start_time),
			"no_transfer": False,
			"layers": layers,
			"n_q": num_q,
			# "g_q": list_gates,
		}
		#print(data)
		data['n_x'] = arch_size
		data['n_y'] = arch_size
		data['n_r'] = arch_size
		data['n_c'] = arch_size
		#codegen = CodeGen(data)
		#program = codegen.builder(no_transfer=False)



