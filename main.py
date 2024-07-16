from naqst import *
from openpyxl import Workbook
import math
import time
from vfsexp import Vf
from SA import find_map_SA

if __name__ == "__main__":

	#qasm input
	path = "Data/qft/"
	#files = os.listdir(path)
	#file_name = 'qft_50.qasm'
	for num_file in range(40, 45):
		file_name = 'qft_{}.qasm'.format(num_file)
		print(file_name)
		#wb = Workbook()
		#ws = wb.active
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
		tmp_graph = nx.Graph()
		tmp_graph.add_edges_from(partition_gates[0])
		initial_map = get_rx_one_mapping(tmp_graph, coupling_graph)
		embeddings.append(map2list(initial_map,num_q))
		print("partition number:", len(partition_gates))
		for i in range(1, len(partition_gates)):
			data = []
			tmp_graph = nx.Graph()
			tmp_graph.add_edges_from(partition_gates[i])
			print(tmp_graph.edges())
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
			print(embeddings[-1])
			print(time2-time1)
		

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


   	#TODO: for the subsequent embeddings, use AOD to move from current embedding to next embedding




