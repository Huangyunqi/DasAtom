from naqst import *
from openpyxl import Workbook
import math
import time
from vfsexp import Vf
from Enola.codegen import CodeGen, global_dict
from Enola.route import QuantumRouter
import json
from SA import find_map_SA

if __name__ == "__main__":

	'''path = "results/tetris/Wstate_cz/map_us/"
	files = os.listdir(path)
	wb = Workbook()
	ws = wb.active
	ws.append(['circuit name', 'Fidelity', 'inserted SWAP', 'gate_cycle'])
	for num_file in range(len(files)):
		file_name = files[num_file]
		if not file_name.endswith('.qasm'):
			continue
		print(file_name)
		cycle_file = file_name+'.txt'
		print(cycle_file)
		Fidelity, swap_count, gate_cycle = compute_fidelity_tetris(cycle_file, file_name, path)
		ws.append([file_name, Fidelity, swap_count, gate_cycle])
	wb.save(path+'tetris_total.xlsx')'''
	#qasm input
	#path = "Data/qft/"
	path_type = 'Tetris_cz'
	path = "Data/{}/circuits/".format(path_type)
	path_embeddings = "Data/{}/Rb2Re4/embeddings/".format(path_type)
	path_partitions = "Data/{}/Rb2Re4/partitions/".format(path_type)
	path_result = "results/yq_test/{}/Rb2Re4/".format(path_type)
	files = os.listdir(path)
	#file_name = 'qft_50.qasm'
	total_wb = Workbook()
	total_ws = total_wb.active
	total_ws.append(['file name','Qubits','CZ_gates', 'fidelity', 'movement_fidelity', 'movement times', 'gate cycles', 'partitions', 'Times'])
	for num_file in range(len(files)):
	#for num_file in [5]:
		#file_name = 'cz_2q_twolocalrandom_indep_qiskit_{}.qasm'.format(num_file+5)
		#file_name = 'cz_2q_qft_{}.qasm'.format(num_file+5)
		file_name = files[num_file]
		print(file_name)
		wb = Workbook()
		ws = wb.active
		log = []
		total_time = time.time()
		cz_circuit = CreateCircuitFromQASM(file_name, path)
	#transform to cz-based circuit
		#cz_circuit = transpile(circuit, basis_gates=['cz', 'rx', 'ry', 'rz', 'h', 't'])
	#cz gates list
		gate_2q_list = get_2q_gates_list(cz_circuit)
	#print(gate_2q_list)
	#obtain corresponding DAG
		_, dag = gates_list_to_QC(gate_2q_list)
		gate_num = len(gate_2q_list)
	#obtain the qubits number
		num_q = qubits_num(gate_2q_list)
		print("Num of gates", gate_num)
		log.append(['Num of gate', gate_num])
		arch_size = math.ceil(math.sqrt(num_q))
		#Rb = math.sqrt(2)
		log.append(['arch_size', 'sqrt(num_q)', arch_size])
		Rb = 2
		log.append(['Rb', '2'])
		r_re = 4 
		log.append(['r_re', '4'])
	#obtain the corresponding coupling_graph 
		coupling_graph = generate_grid_with_Rb(arch_size,arch_size, Rb)

	#obtain the gates partition
		time_part = time.time()
		#ini_map = qasm_to_map('results/initial_map/'+file_name)
		partition_gates = parition_from_DAG(dag, coupling_graph)
		write_data(partition_gates, path_partitions, file_name.removesuffix(".qasm")+'.txt')
		#partition_gates = partition_from_ini(dag, coupling_graph, ini_map)
		#partition_gates = read_data(path_partitions, file_name.removesuffix(".qasm")+'.txt')
		time_part1 = time.time()
		print("partition time is, ",time_part1-time_part)
		log.append(["partition time", time_part1-time_part])
		#ws.append(["partition time", time_part1-time_part])
	#print("------------------------------")
		part_gate_num = 0
		for gates in partition_gates:
	#		print("------")
			part_gate_num += len(gates)
	#	print(gates)
		print("final num is:", part_gate_num)


    #for each partition, find a proper embedding
		time_embed = time.time()
		embeddings = get_embeddings(partition_gates, coupling_graph, num_q)
		#embeddings = read_data(path_embeddings, file_name.removesuffix(".qasm")+'.txt')
		time_embed1 = time.time()
		print("partition number:", len(partition_gates))
		log.append(["find embeddings time", time_embed1-time_embed])

		write_data(embeddings, path_embeddings, file_name.removesuffix(".qasm")+'.txt')
		parallel_gates = []
		time_paral = time.time()
		for i in range(len(partition_gates)):
			gates = get_parallel_gates(partition_gates[i], coupling_graph, embeddings[i], r_re)
			parallel_gates.append(gates)
		time_paral1 = time.time()
		log.append(["find parallel_gates time", time_paral1-time_paral])


		#route = QuantumRouter(num_q, embeddings, partition_gates, [arch_size, arch_size])
		#route.run()
		#all_movements = route.momvents
		#print(all_movements)

		window = False
		window_size = 1000
		routing_strategy = "maximalis"
		layers = []
		initial_map = map_to_layer(embeddings[0])
		initial_map["gates"] = gate_in_layer(partition_gates[0])
		layers.append(initial_map)
		all_movements = []
		total_paralled = []
		for num in range(len(embeddings) - 1):
			log.append([str(embeddings[num])])
			for gates in parallel_gates[num]:
				log.append([str(gates[it]) for it in range(len(gates))])
				total_paralled.append(gates)

			current_map = embeddings[num]
			next_map = embeddings[num + 1]
			last_layer = map_to_layer(current_map)
			next_layer = map_to_layer(next_map)
		
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
				all_movements.append(movement_result)
				log.append([str(movement_result[it]) for it in range(len(movement_result))])
				layers.append(new_layer)
				for i in range(num_q):
					if new_layer["qubits"][i]["a"] == 1:
						last_layer["qubits"][i] = next_layer["qubits"][i]
				
			if movements:
				for qubit in movements:
					move = movements[qubit]
					all_movements.append([[qubit, [move[0],move[1]],[move[2],move[3]]]])
					log.append([str([qubit, [move[0],move[1]], [move[2],move[3]]])])
					for qubit_ in last_layer["qubits"]:
						if qubit_["id"] == qubit:
							qubit_["a"] = 1
				layers.append(last_layer)
			layers[-1]["gates"] = gate_in_layer(partition_gates[num+1])
		# layers.append(next_layer)
		if len(partition_gates) > 1:
			log.append([str(embeddings[num+1])])
			for gates in parallel_gates[num+1]:
				log.append([str(gates[it]) for it in range(len(gates))])
				total_paralled.append(gates)
		else:
			log.append([str(embeddings[0])])
			for gates in parallel_gates[0]:
				log.append([str(gates[it]) for it in range(len(gates))])
				total_paralled.append(gates)
		t_idle, Fidelity, move_fidelity = compute_fidelity(total_paralled, all_movements, num_q, gate_num)
		print("Fidelity is:", Fidelity)
		log.append(["Fidelity:", Fidelity])
		log.append(["t_idle:", t_idle])
		log.append(["move_fidelity", move_fidelity])
		log.append(["Movement times", len(all_movements)])
		log.append(["parallel times", len(total_paralled)])
		log.append(["partitions", len(embeddings)])
		total_time1 = time.time()
		log.append(["total time:", total_time1-total_time])
		para = set_parameters(True)
		log_para = []
		for key, value in para.items():
			log_para.append(str(key))
			log_para.append(str(value))
		log.append(log_para)
		for item in log:
			#print(item)
			ws.append(item)

		total_ws.append([file_name, num_q, gate_num, Fidelity, move_fidelity, len(all_movements), len(total_paralled), len(embeddings), total_time1-total_time])
		save_file = path_result+'{}_rb{}_archsize{}_mini_dis.xlsx'.format(file_name, Rb, arch_size)
		print(save_file)
		wb.save(save_file)

		data = {
		# "runtime": float(time.time() - start_time),
		"no_transfer": False,
		"layers": layers,
		"n_q": num_q,
		"g_q": gate_2q_list,
		}
 
		global_dict['full_code'] = True

		data['n_x'] = arch_size
		data['n_y'] = arch_size
		data['n_r'] = arch_size
		data['n_c'] = arch_size
	# print("#layers: {}".format(len(data["layers"])))
	# t_s = time.time()
		codegen = CodeGen(data)
		program = codegen.builder(no_transfer=False)
		program = program.emit_full()
	para = set_parameters(True)
	log_para = []
	for key, value in para.items():
		log_para.append(str(key))
		log_para.append(str(value))
	total_ws.append(log_para)
	total_wb.save(path_result+'total_tet_our_mini_dis_arch1.xlsx')
		#if global_dict["full_code"]:
		#	with open(f"results/test_{num_q}_{0}_code_full.json", 'w') as f:
		#		json.dump(program, f)
		#		for instruction in program:
		#			instruction["state"] = {}
    # optional
    # run following command in terminal:
    # python Enola/animation.py f"Data/test_{num_q}_{0}_code_full.json" --dir "./Data/"'''



