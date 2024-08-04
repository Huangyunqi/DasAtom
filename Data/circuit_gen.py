from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit.qasm2.export import dump

import os

def CreateCircuitFromQASM(file, path):
	#print(path+file)
	QASM_file = open(path + file, 'r')
	iter_f = iter(QASM_file)
	QASM = ''
	for line in iter_f: 
		QASM = QASM + line
	cir = QuantumCircuit.from_qasm_str(QASM)
	QASM_file.close    
	return cir

def create_ghz_circuit(n_beg, n_end, path):
	for n in range(n_beg, n_end):
		ghz_circuit = QuantumCircuit(n)
		ghz_circuit.h(0)
		for qubit in range(n-1):
			ghz_circuit.cx(qubit, qubit+1)
		file_name = 'GHZ_{}.qasm'.format(n)
		dump(ghz_circuit, path+file_name)

def transform_cz_only(file_name, path):
	cir = CreateCircuitFromQASM(file_name, path)
	cz_circuit = transpile(cir, basis_gates=['cz', 'rx', 'ry', 'rz', 'h', 't'])
	n = cir.num_qubits
	#print(cz_circuit)
	cz_only = QuantumCircuit(n)
	instruction = cz_circuit.data
	for ins in instruction:
		if ins.operation.num_qubits == 2:
			cz_only.cz(ins.qubits[0]._index, ins.qubits[1]._index)
	#print(cz_only)
	return cz_only
if __name__ == "__main__":

	path = 'qft/'
	#create_ghz_circuit(5, 21, path)

	files = os.listdir(path)
	for file_name in files:
		cz_cir = transform_cz_only(file_name, path)
		dump(cz_cir, 'qft_cz/cz_2q_'+file_name)


