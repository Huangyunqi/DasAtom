from qiskit import qasm2, transpile, QuantumCircuit, QuantumRegister
from qiskit.qasm2.export import dump
import os

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

def limited_circuit(cir):
    C = []
    for gate in cir:
        if gate[0].name != 'cz': 
            continue
        qubits = [q._index for q in gate[1]]
        C.append(qubits)
    max_qubit_index = max(max(gate) for gate in C)
    num_qubits = max_qubit_index + 1
    circ = QuantumCircuit(num_qubits)
    for cz in C:
        circ.cz(cz[0], cz[1])
    return circ

if __name__ == "__main__":

	path = "examples/"
	files = os.listdir(path)
	for num_file in range(len(files)):
		file_name = files[num_file]
		circuit = CreateCircuitFromQASM(file_name, path)
		cz_circuit = transpile(circuit, basis_gates=['cz', 'rx', 'ry', 'rz', 'h', 't'])
		cz_2q_cir = limited_circuit(cz_circuit)
		new_file = "examples_cz/cz_2q_{}".format(file_name)
		dump(cz_2q_cir, new_file)

