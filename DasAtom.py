import os
import time
import math
from openpyxl import Workbook
import warnings
from Enola.route import QuantumRouter
from naqst import *
import argparse

class DasAtom:
    def __init__(self, bench_name: str, circuit_folder: str,
                radius_interaction = 2,
                save_folder = None,
                embeddings_from_read=False, save_embeddings=True, save_cir_res=True, save_bench_res=True):

        self.bench_name = bench_name
        self.Rb = radius_interaction
        self.Re = 2*self.Rb

        if not os.path.exists(circuit_folder):
            raise FileNotFoundError(f"The specified directory does not exist: {circuit_folder}")
        self.cir_folder = circuit_folder
        if not save_folder:
            save_folder = f"res/{self.bench_name}"
        if os.path.exists(save_folder):
            warnings.warn(f"The results for '{self.bench_name}' may be overwritten. \
                To avoid this, consider using a different result folder before 'process_files' with the function 'modify_result_folder'.")
        self.res_folder = save_folder
        os.makedirs(self.res_folder,exist_ok=True)

        self.files = [f for f in os.listdir(self.cir_folder) if f.endswith('.qasm')]
        self.embeddings_from_read = embeddings_from_read
        self.save_embeddings = save_embeddings
        self.save_cir_res = save_cir_res
        self.save_bench_res = save_bench_res


    def modify_result_folder(self,save_folder:str):
        if not os.path.exists(save_folder):
            self.res_folder = save_folder
            os.makedirs(self.res_folder)
        else:
            print(f"Try other folder path, {save_folder} exists")

    def process_files(self, file_indices=None):
        self.path_result = os.path.join(self.res_folder,f"Rb{self.Rb}Re{self.Re}")
        self.path_embeddings = os.path.join(self.path_result,"embeddings")
        self.path_partitions = os.path.join(self.path_result,"partitions")
        os.makedirs(self.path_embeddings,exist_ok=True)
        os.makedirs(self.path_partitions,exist_ok=True)
        # init total_wb
        self.total_wb = Workbook()
        self.total_ws = self.total_wb.active
        self.total_ws.append(['file name', 'Qubits', 'CZ_gates', 'depth', 'fidelity', 'movement_fidelity', 'movement times', 'gate cycles', 'partitions', 'Times'])

        if not file_indices:
            file_indices = range(len(self.files))

        for num_file in file_indices:
            file_name = self.files[num_file]
            print(f"{file_name} running")
            self._process_file(file_name)

        para = set_parameters(True)
        log_para = []
        for key, value in para.items():
            log_para.append(str(key))
            log_para.append(str(value))
        self.total_ws.append(log_para)
        if self.save_bench_res:
            self.total_wb.save(os.path.join(self.path_result,f'{self.bench_name}_big.xlsx'))


    def _process_file(self, file_name):
        wb = Workbook()
        ws = wb.active
        self.log = []
        total_time = time.time()

        cz_circuit = CreateCircuitFromQASM(file_name,self.cir_folder)
        gate_2q_list = get_2q_gates_list(cz_circuit)
        cir, dag = gates_list_to_QC(gate_2q_list)

        num_q, gate_num, arch_size= self._calculate_architecture(gate_2q_list)

        coupling_graph = self._generate_coupling_graph(arch_size)

        partition_gates = self._get_partition_gates(file_name, coupling_graph, dag)
        embeddings, arch_size = self._get_embeddings(file_name, partition_gates, coupling_graph, num_q, arch_size)

        parallel_gates, all_movements = self._get_parallel_movements(num_q, partition_gates, embeddings, coupling_graph, arch_size)

        t_idle, Fidelity, move_fidelity = self._compute_fidelity(parallel_gates, all_movements, num_q, gate_num)

        self.log.append(["total time:", time.time() - total_time])
        for item in self.log:
            ws.append(item)

        save_file_name = os.path.join(self.path_result,f'{file_name}_rb{self.Rb}_archSize{arch_size}_mini_dis.xlsx')
        if self.save_cir_res:
            wb.save(save_file_name)


        self.total_ws.append([file_name, num_q, gate_num, cir.depth(), Fidelity, move_fidelity, len(all_movements), len(parallel_gates), len(embeddings), total_time])

        return

    def _calculate_architecture(self, gate_2q_list):
        gate_num = len(gate_2q_list)
        num_q = qubits_num(gate_2q_list)
        arch_size = math.ceil(math.sqrt(num_q))

        self.log.append(['Num of gate', gate_num])
        self.log.append(['arch_size', 'sqrt(num_q)', arch_size])
        self.log.append(['Rb', self.Rb])
        self.log.append(['r_re', self.Re])

        return num_q, gate_num, arch_size

    def _generate_coupling_graph(self, arch_size):
        return generate_grid_with_Rb(arch_size, arch_size, self.Rb)

    def _get_partition_gates(self, file_name, coupling_graph, dag):
        if self.embeddings_from_read:
            return read_data(self.path_partitions, file_name.removesuffix(".qasm") + '.txt')
        else:
            time_part = time.time()
            partition_gates = partition_from_DAG(dag, coupling_graph)
            if self.save_embeddings:
                write_data(partition_gates, self.path_partitions, file_name.removesuffix(".qasm") + 'part.txt')
            self.log.append(["partition time", time.time() - time_part])
            return partition_gates

    def _get_embeddings(self, file_name, partition_gates, coupling_graph, num_q, arch_size):
        if self.embeddings_from_read:
            embeddings = read_data(self.path_embeddings, file_name.removesuffix(".qasm") + '.txt')
        else:
            time_embed = time.time()
            embeddings, extend_pos = get_embeddings(partition_gates, coupling_graph, num_q, arch_size, self.Rb)
            if self.save_embeddings:
                write_data(embeddings, self.path_embeddings, file_name.removesuffix(".qasm") + 'emb.txt')
            self.log.append(["find embeddings time", time.time() - time_embed])
            if len(extend_pos) != 0:
                self.log.append(["extend graph times", len(extend_pos)])
                self.log.append(extend_pos)
                arch_size += len(extend_pos)

        return embeddings, arch_size

    def _get_parallel_movements(self,num_q, partition_gates, embeddings, coupling_graph, arch_size):
        parallel_gates = []
        all_movements = []

        route = QuantumRouter(num_q, embeddings, partition_gates, [arch_size, arch_size])
        route.run()

        for i in range(len(partition_gates)):
            gates = get_parallel_gates(partition_gates[i], coupling_graph, embeddings[i], self.Re)
            parallel_gates.append(gates)

        for num in range(len(embeddings) - 1):
            for gates in parallel_gates[num]:
                self.log.append([str(gates[it]) for it in range(len(gates))])
            for para_moves in route.movement_list[num]:
                self.log.append([str(para_moves[it]) for it in range(len(para_moves))])
                all_movements.append(para_moves)

        return parallel_gates, all_movements

    def _compute_fidelity(self, parallel_gates, all_movements, num_q, gate_num):
        t_idle, Fidelity, move_fidelity = compute_fidelity(parallel_gates, all_movements, num_q, gate_num)
        self.log.append(["Fidelity:", Fidelity])
        self.log.append(["t_idle:", t_idle])
        self.log.append(["move_fidelity", move_fidelity])
        self.log.append(["Movement times", len(all_movements)])
        self.log.append(["parallel times", len(parallel_gates)])

        return t_idle, Fidelity, move_fidelity


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize DasAtom class with the given parameters.")

    parser.add_argument("bench_name", type=str, help="Name of the benchmark.")
    parser.add_argument("circuit_folder", type=str, help="Path to the folder containing circuit data.")
    parser.add_argument("--radius_interaction", type=int, default=2, help="Radius of interaction (default: 2).")
    parser.add_argument("--save_folder", type=str, help="Folder to save results.")
    parser.add_argument("--embeddings_from_read", action="store_true", help="Whether to read embeddings from file.")
    parser.add_argument("--no_save_embeddings", action="store_false", dest="save_embeddings", help="Do not save embeddings.")
    parser.add_argument("--no_save_cir_res", action="store_false", dest="save_cir_res", help="Do not save circuit results.")
    parser.add_argument("--no_save_bench_res", action="store_false", dest="save_bench_res", help="Do not save benchmark results.")

    args = parser.parse_args()

    das_atom = DasAtom(
        bench_name=args.bench_name,
        circuit_folder=args.circuit_folder,
        radius_interaction=args.radius_interaction,
        save_folder=args.save_folder,
        embeddings_from_read=args.embeddings_from_read,
        save_embeddings=args.save_embeddings,
        save_cir_res=args.save_cir_res,
        save_bench_res=args.save_bench_res
    )
    das_atom.process_files()
