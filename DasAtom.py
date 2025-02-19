import os
import time
import math
from openpyxl import Workbook
import warnings
from Enola.route import QuantumRouter, compatible_2D
from DasAtom_fun import *
import argparse

class SingleFileProcessor:
    """
    A helper class responsible for processing a single QASM file. This class:
        - Reads the circuit from QASM.
        - Computes gate lists and partitions.
        - Retrieves/Generates embeddings.
        - Computes parallel gates and necessary qubit-movement operations.
        - Calculates fidelity and time metrics.
        - Saves per-file results.
    """

    def __init__(
        self,
        qasm_filename: str,
        circuit_folder: str,
        benchmark_name: str,
        interaction_radius: int,
        extended_radius: int,
        result_path: str,
        embeddings_path: str,
        partitions_path: str,
        read_embeddings: bool,
        save_partitions_and_embeddings: bool,
        save_circuit_results: bool,
        save_benchmark_results: bool,
        if_verify:bool = False
    ):
        """
        Initialize the processor with file-specific and benchmark-wide parameters.

        :param qasm_filename: Name of the QASM file to process (e.g., 'circuit_14.qasm').
        :param circuit_folder: Directory containing the QASM file.
        :param benchmark_name: The benchmark name (used for naming output files).
        :param interaction_radius: The interaction radius (Rb).
        :param extended_radius: The extended interaction radius (2 * Rb).
        :param result_path: Path to the parent results folder.
        :param embeddings_path: Path to the folder where embeddings are read/saved.
        :param partitions_path: Path to the folder where partitions are read/saved.
        :param read_embeddings: Whether to read embeddings from existing files (instead of computing).
        :param save_partitions_and_embeddings: Whether to save newly created partitions/embeddings to disk.
        :param save_circuit_results: Whether to save circuit-level results (xlsx).
        :param save_benchmark_results: Whether to save the overall benchmark-level results.
        :param if_verify: Whether to verify the whole process.
        """
        self.qasm_filename = qasm_filename
        self.circuit_folder = circuit_folder
        self.benchmark_name = benchmark_name
        self.interaction_radius = interaction_radius
        self.extended_radius = extended_radius
        self.result_path = result_path
        self.embeddings_path = embeddings_path
        self.partitions_path = partitions_path
        self.read_embeddings = read_embeddings
        self.save_partitions_and_embeddings = save_partitions_and_embeddings
        self.save_circuit_results = save_circuit_results
        self.save_benchmark_results = save_benchmark_results
        self.if_verify = if_verify
        self.qubit_num = 0

        # Used to store logs for the final XLSX per file
        self.file_process_log = []
        
        
    def validate_embeddings(self, embeddings):
        """
        Validate consistency and uniqueness of embeddings.
        
        :param embeddings: A three-layer nested list where:
            - First layer: Different embeddings.
            - Second layer: The map of qubits.
            - Third layer: (x, y) coordinates of the ith qubit.
        
        :return: True if validation passes, otherwise raises an assertion error.
        """

        for idx, embedding in enumerate(embeddings):
            # Ensure all embeddings have the expected qubit count
            assert len(embedding) == self.qubit_num, (
                f"Inconsistent qubit count at embedding index {idx}: "
                f"expected {self.qubit_num}, but got {len(embedding)}."
            )

            # Ensure all qubit positions are unique
            unique_positions = set(embedding)
            assert len(embedding) == len(unique_positions), (
                f"Duplicate qubit positions found in embedding index {idx}. "
                f"Expected unique locations but found duplicates: {embedding}."
            )

        return True  # If no assertion fails, embeddings are valid.
        
    def validate_partition_embedding(self, partitioned_gates, embeddings) -> bool:
        """
        Verify the partition.
        
        :param partitioned_gates: A three-layer nested list where:
            - First layer: Different execution partitions.
            - Second layer: Groups of gates executed together.
            - Third layer: Individual gate operations.
        :param embeddings: A three-layer nested list where:
            - First layer: Different embeddings.
            - Second layer: The map of qubits.
            - Third layer: (x, y) coordinates of the ith qubit.
        :return: True if verification passes, otherwise raises an assertion error.
        """
        
        # Ensure the number of execution partitions matches the number of embeddings
        assert len(partitioned_gates) == len(embeddings), (
            f"Mismatch in partition layers: expected {len(embeddings)}, "
            f"but got {len(partitioned_gates)}."
        )

        # Verify each gate respects the embedding and interaction constraints
        for layer_idx, execution_layer in enumerate(partitioned_gates):
            embedding = embeddings[layer_idx]  # Get the corresponding embedding for this layer
            
            for gate in execution_layer:
                
                q0, q1 = gate
                
                # Ensure gate qubits are within bounds
                assert 0 <= q0 < self.qubit_num and 0 <= q1 < self.qubit_num, (
                    f"Invalid qubit indices: q0={q0}, q1={q1}. "
                    f"Must be in range [0, {self.qubit_num - 1})."
                )

                # Get qubit positions
                loc_q0 = embedding[q0]
                loc_q1 = embedding[q1]

                # Ensure qubits respect interaction distance
                distance = euclidean_distance(loc_q0, loc_q1)
                assert distance <= self.interaction_radius, (
                    f"Qubits {q0} and {q1} exceed interaction radius. "
                    f"Distance = {distance}, Max allowed = {self.interaction_radius}."
                )

        return True  # If no assertion fails, the partition is valid.

    def validate_movements(self, embeddings, movements_list) -> bool:
        """
        Validate qubit movements between embeddings.
        
        :param embeddings: A three-layer nested list where:
            - First layer: Different embeddings.
            - Second layer: The map of qubits.
            - Third layer: (x, y) coordinates of the ith qubit.
            
        :param movements_list: A three-layer nested list where:
            - First layer: Different embeddings.
            - Second layer: The move sequences during this movement.
            - Third layer: [qubit, (origin_x, origin_y), (next_x, next_y)].
        
        :return: True if validation passes, otherwise raises an assertion error.
        """

        # Ensure that the number of movements aligns with embeddings (n movements for n+1 embeddings)
        assert len(embeddings) == len(movements_list) + 1, (
            f"Mismatch in movement count: expected {len(embeddings) - 1}, "
            f"but got {len(movements_list)}."
        )

        # Validate each movement step
        for i, movement in enumerate(movements_list):
            current_embedding = embeddings[i]
            next_embedding = embeddings[i + 1]

            for sequence in movement:
                for j, (qubit, (ox, oy), (nx, ny)) in enumerate(sequence):
                    move_j = [ox, oy, nx, ny]

                    # Ensure qubits are moving from and to the correct locations
                    assert (ox, oy) == current_embedding[qubit], (
                        f"Qubit {qubit} mismatch: expected {current_embedding[qubit]}, but got ({ox}, {oy})."
                    )
                    assert (nx, ny) == next_embedding[qubit], (
                        f"Qubit {qubit} mismatch after move: expected {next_embedding[qubit]}, but got ({nx}, {ny})."
                    )

                    # Check compatibility of moves within the same sequence
                    for k in range(j + 1, len(sequence)):
                        other_move = sequence[k]
                        compatible_2D(move_j, [other_move[1][0], other_move[1][1], other_move[2][0], other_move[2][1]])

        return True  # If no assertion fails, movements are valid.

    def validate_parallel_gates(self, parallel_gate_groups, embeddings) -> bool:
        """
        Validate parallel gate execution based on embeddings.
        
        :param parallel_gate_groups: A four-layer nested list where:
            - First layer: Different embedding maps.
            - Second layer: Different gate layers.
            - Third layer: Parallel gates.
            - Fourth layer: [i, j] pairs representing individual gate operations.
        
        :param embeddings: A three-layer nested list where:
            - First layer: Different embeddings.
            - Second layer: The map of qubits.
            - Third layer: (x, y) coordinates of the ith qubit.
        
        :return: True if validation passes, otherwise raises an assertion error.
        """
        
        for layer_idx, gate_group in enumerate(parallel_gate_groups):
            embedding = embeddings[layer_idx]  # Get the corresponding embedding for this layer
            
            for execution_layer in gate_group:
                locs = []
                locs = []
                
                for gate in execution_layer:
                    q0, q1 = gate
                    
                    # Ensure gate qubits are within valid range
                    assert 0 <= q0 < self.qubit_num and 0 <= q1 < self.qubit_num, (
                        f"Invalid qubit indices: q0={q0}, q1={q1}. "
                        f"Must be in range [0, {self.qubit_num - 1}]."
                    )
                    
                    # Get qubit positions
                    locs.append(embedding[q0])
                    locs.append(embedding[q1])
                
                # Ensure all qubits in `loc` are sufficiently spaced
                for i in range(len(locs)):
                    for j in range(i+2-i%2, len(locs)):
                        dist = euclidean_distance(locs[i], locs[j])
                        assert dist >= self.extended_radius, (
                            f"Qubit overlap issue in layer {layer_idx}: "
                            f"Distance between {locs[i]} and {locs[j]} = {dist}, "
                            f"but should be >= {self.extended_radius}."
                        )
        
        return True  # If no assertion fails, gates are valid.
   
    def process_qasm_file(self):
        """
        Main entry point to process the single QASM file. This function:
            1. Builds the circuit from the QASM file.
            2. Partitions the circuit and obtains embeddings.
            3. Generates parallel gates and qubit-movement sequences.
            4. Computes fidelity metrics.
            5. Saves logs to a per-file Excel sheet (if configured).

        :return: A list of metrics to be appended as a row in the main (benchmark-wide) workbook.
        """
        wb = Workbook()
        ws = wb.active
        start_time = time.time()

        # 1) Create circuit from QASM, then extract 2-qubit gates and DAG
        qasm_circuit = CreateCircuitFromQASM(self.qasm_filename, self.circuit_folder)
        two_qubit_gates_list = get_2q_gates_list(qasm_circuit)
        assert two_qubit_gates_list, f"a wrong circuit which have no cz in {self.qasm_filename}"
        qc_object, dag_object = gates_list_to_QC(two_qubit_gates_list)

        # 2) Determine key architecture parameters
        num_qubits, num_cz_gates, grid_size = self._compute_architecture_parameters(two_qubit_gates_list)
        self.qubit_num = num_qubits

        # 3) Generate coupling graph based on the interaction radius
        coupling_graph = self._generate_coupling_graph(grid_size)

        # 4) Get or create partitions
        partitioned_gates = self._retrieve_or_generate_partitions(self.qasm_filename, coupling_graph, dag_object)

        # 5) Get or create embeddings
        embeddings, grid_size = self._retrieve_or_generate_embeddings(
            self.qasm_filename,
            partitioned_gates,
            coupling_graph,
            num_qubits,
            grid_size
        )

        # 6) Generate parallel gates and all movement operations
        parallel_gates, movements_list, merged_parallel_gates = self._compute_gates_and_movements(
            num_qubits,
            partitioned_gates,
            embeddings,
            coupling_graph,
            grid_size
        )

        # 7) Compute fidelity/time metrics
        total_time_now = time.time()
        idle_time, fidelity, move_fidelity, total_runtime, num_transfers, num_moves, total_move_distance = compute_fidelity(
            merged_parallel_gates,
            movements_list,
            num_qubits,
            num_cz_gates
        )

        # 8) Log final stats for this file
        self.file_process_log.append(["Total processing time", total_time_now - start_time])
        self.file_process_log.append(["Original circuit depth", qc_object.depth()])
        self.file_process_log.append(["Fidelity", fidelity])
        self.file_process_log.append(["Idle time", idle_time])
        self.file_process_log.append(["Movement fidelity", move_fidelity])
        self.file_process_log.append(["Movement operations", len(movements_list)])
        self.file_process_log.append(["Parallel gate groups", len(merged_parallel_gates)])
        self.file_process_log.append(["Number of partitions", len(embeddings)])
        self.file_process_log.append(["Num of qubit moves (transfers)", num_transfers])
        self.file_process_log.append(["Num of final re-locations (moves)", num_moves])
        self.file_process_log.append(["Total move distance", total_move_distance])
        self.file_process_log.append(["Total run time", total_time_now - start_time])

        # 9) Optionally save a per-file XLSX
        save_file_name = os.path.join(
            self.result_path,
            f'{self.qasm_filename}_rb{self.interaction_radius:.3g}.xlsx'
        )
        for item in self.file_process_log:
            ws.append(item)
        if self.save_circuit_results:
            wb.save(save_file_name)

        # 10) Return the row of aggregated stats for the main (benchmark-wide) workbook
        return [
            self.qasm_filename,
            num_qubits,
            num_cz_gates,
            qc_object.depth(),
            fidelity,
            move_fidelity,
            len(movements_list),
            num_moves * 4,           # num of transfer
            num_moves,
            total_move_distance,
            len(merged_parallel_gates),
            len(embeddings),
            (total_time_now - start_time),
            total_runtime,
            idle_time
        ]

    def _compute_architecture_parameters(self, two_qubit_gates_list):
        """
        Compute the number of qubits, the number of gates, and an initial grid dimension
        for the architecture based on the QASM file's gate set.

        :param two_qubit_gates_list: The list of extracted 2-qubit gates.
        :return: (num_qubits, num_cz_gates, grid_size)
        """
        num_cz_gates = len(two_qubit_gates_list)
        num_qubits = get_qubits_num(two_qubit_gates_list)
        grid_size = math.ceil(math.sqrt(num_qubits))

        self.file_process_log.append(["Number of CZ gates", num_cz_gates])
        self.file_process_log.append(["Initial grid size (sqrt(num_qubits))", grid_size])
        self.file_process_log.append(["Interaction radius (Rb)", self.interaction_radius])
        self.file_process_log.append(["Extended radius (Re)", self.extended_radius])

        return num_qubits, num_cz_gates, grid_size

    def _generate_coupling_graph(self, grid_size):
        """
        Create a 2D grid-based coupling graph based on the specified grid size
        and the interaction radius.

        :param grid_size: The number of rows/columns in the square grid.
        :return: A graph representing qubit coupling.
        """
        return generate_grid_with_Rb(grid_size, grid_size, self.interaction_radius)

    def _retrieve_or_generate_partitions(self, filename, coupling_graph, dag_object):
        """
        Retrieve precomputed partitions from JSON if read_embeddings is True,
        otherwise partition the circuit's DAG and optionally save to JSON.

        :param filename: Name of the QASM file (without path).
        :param coupling_graph: Graph of qubit couplings.
        :param dag_object: DAG representation of the circuit.
        :return: A list of partitioned gates.
        """
        if self.read_embeddings:
            return read_data(
                self.partitions_path,
                filename.removesuffix(".qasm") + '.json'
            )
        else:
            start_partition_time = time.time()
            partitioned_gates = partition_from_DAG(dag_object, coupling_graph)
            self.file_process_log.append(["Partitioning time", time.time() - start_partition_time])

            if self.save_partitions_and_embeddings:
                write_data_json(
                    partitioned_gates,
                    self.partitions_path,
                    filename.removesuffix(".qasm") + 'part.json'
                )
            return partitioned_gates

    def _retrieve_or_generate_embeddings(
        self,
        filename,
        partitioned_gates,
        coupling_graph,
        num_qubits,
        grid_size
    ):
        """
        Retrieve or compute embeddings for each partition. If read_embeddings
        is True, read from JSON. Otherwise, compute embeddings and optionally save.

        :param filename: QASM file name (string).
        :param partitioned_gates: A list of partitioned gates (from partition_from_DAG).
        :param coupling_graph: Qubit coupling graph.
        :param num_qubits: Number of qubits in the circuit.
        :param grid_size: Current grid dimension.
        :return: (embeddings, potentially updated grid_size)
        """
        if self.read_embeddings:
            embeddings = read_data(
                self.embeddings_path,
                filename.removesuffix(".qasm") + '.json'
            )
            return embeddings, grid_size
        else:
            start_embed_time = time.time()
            embeddings, extended_positions = get_embeddings(
                partitioned_gates,
                coupling_graph,
                num_qubits,
                grid_size,
                self.interaction_radius
            )
            self.file_process_log.append(["Embedding computation time", time.time() - start_embed_time])

            if self.save_partitions_and_embeddings:
                write_data_json(
                    embeddings,
                    self.embeddings_path,
                    filename.removesuffix(".qasm") + 'emb.json'
                )

            # If graph was extended, reflect this in the grid_size
            if extended_positions:
                self.file_process_log.append(["Graph extension count", len(extended_positions)])
                self.file_process_log.append(["Extended positions", extended_positions])
                grid_size += len(extended_positions)

            if self.if_verify:
                try:
                    self.validate_embeddings(embeddings)
                    self.validate_partition_embedding(partitioned_gates, embeddings)
                except AssertionError as e:
                    print(f"Verification failed: {e}")  # Or use logging
                    raise
                except Exception as e:
                    print(f"Unexpected error during verification: {e}")
                    raise

            return embeddings, grid_size

    def _compute_gates_and_movements(self, num_qubits, partitioned_gates, embeddings, coupling_graph, grid_size):
        """
        Use the QuantumRouter to determine how to move qubits between partitions.
        Also compute the parallel gates for each partition based on the extended radius.

        :param num_qubits: Number of qubits in the circuit.
        :param partitioned_gates: Gates partitioned by circuit stage.
        :param embeddings: Embeddings for each partition.
        :param coupling_graph: Grid-based qubit coupling graph.
        :param grid_size: Dimensions of the square grid.
        :return: (list of parallel gate groups, list of all movement operations, merged list of parallel gates)
        """
        parallel_gate_groups = []
        movement_operations = []
        merged_parallel_gates = []

        # Generate the parallel gates for each partition
        for i in range(len(partitioned_gates)):
            gates = get_parallel_gates(
                partitioned_gates[i],
                coupling_graph,
                embeddings[i],
                self.extended_radius
            )
            parallel_gate_groups.append(gates)

        if self.if_verify:
            try:
                self.validate_parallel_gates(parallel_gate_groups, embeddings)
            except AssertionError as e:
                print(f"Verification failed: {e}")  # Or use logging
                raise
            except Exception as e:
                print(f"Unexpected error during verification: {e}")
                raise

        # QuantumRouter: figure out the qubit re-locations from partition N to N+1
        router = QuantumRouter(
            num_qubits, embeddings, parallel_gate_groups, [grid_size, grid_size]
        )
        router.run()
        if self.if_verify:
            try:
                self.validate_movements(embeddings, router.movement_list)
            except AssertionError as e:
                print(embeddings, router.movement_list)
                print(f"Verification failed: {e}")  # Or use logging
                raise
            except Exception as e:
                print(f"Unexpected error during verification: {e}")
                raise 
        router.save_program(
            os.path.join(self.embeddings_path, f"{self.benchmark_name}_{num_qubits}.json")
        )
        
        # Append parallel gates and movement sequences
        for i in range(len(embeddings) - 1):
            # Log parallel gate group for partition i
            for g_list in parallel_gate_groups[i]:
                self.file_process_log.append([str(g) for g in g_list])
                merged_parallel_gates.append(g_list)

            # Movement from partition i to partition i+1
            for move_group in router.movement_list[i]:
                self.file_process_log.append([str(m) for m in move_group])
                movement_operations.append(move_group)

        # The last partition (which doesn't need to move to a next partition)
        if len(partitioned_gates) > 0:
            self.file_process_log.append([str(embeddings[-1])])
            for g_list in parallel_gate_groups[-1]:
                self.file_process_log.append([str(g) for g in g_list])
                merged_parallel_gates.append(g_list)

        return parallel_gate_groups, movement_operations, merged_parallel_gates


class DasAtom:
    """
    Main class to handle multiple QASM files (i.e., the entire benchmark).
    Responsibilities:
        - Storing benchmark-level configurations.
        - Iterating over all QASM files in the input directory.
        - Invoking SingleFileProcessor for each QASM file.
        - Maintaining a master Excel workbook of aggregated results.
    """
    def __init__(
        self,
        benchmark_name: str,
        circuit_folder: str,
        interaction_radius: int = 2,
        results_folder: str = None,
        read_embeddings: bool = False,
        save_partitions_and_embeddings: bool = True,
        save_circuit_results: bool = True,
        save_benchmark_results: bool = True,
        if_verify: bool = False
    ):
        """
        Initialize the multi-file processor with user-provided settings.

        :param benchmark_name: Name of the benchmark (used in output naming).
        :param circuit_folder: Path containing the QASM files to process.
        :param interaction_radius: The interaction radius (Rb).
        :param results_folder: The parent folder where results are stored (defaults to 'res/{benchmark_name}').
        :param read_embeddings: If True, read existing embeddings/partitions from disk.
        :param save_partitions_and_embeddings: If True, save newly computed partitions/embeddings to JSON.
        :param save_circuit_results: If True, save per-circuit XLSX logs.
        :param save_benchmark_results: If True, save a master XLSX for all circuits.
        :param if_verify: Whether to verify the whole process.
        """
        self.benchmark_name = benchmark_name
        self.interaction_radius = interaction_radius
        self.extended_radius = 2 * self.interaction_radius

        assert os.path.exists(circuit_folder), f"Directory not found: {circuit_folder}"
        self.circuit_folder = circuit_folder

        # Default results folder: 'res/{benchmark_name}'
        if results_folder is None:
            results_folder = f"res/{self.benchmark_name}"
        if os.path.exists(results_folder):
            warnings.warn(
                f"The results for '{self.benchmark_name}' may be overwritten in: {results_folder}. "
                f"Consider using a different folder to preserve existing results."
            )
        self.results_folder = results_folder
        os.makedirs(self.results_folder, exist_ok=True)

        # Collect all .qasm files
        qasm_files = [f for f in os.listdir(self.circuit_folder) if f.endswith('.qasm')]
        self.qasm_files = sorted(qasm_files, key=self._extract_numeric_suffix)

        self.read_embeddings = read_embeddings
        self.save_partitions_and_embeddings = save_partitions_and_embeddings
        self.save_circuit_results = save_circuit_results
        self.save_benchmark_results = save_benchmark_results
        self.if_verify = if_verify

    @staticmethod
    def _extract_numeric_suffix(filename: str):
        """
        Extract a numeric suffix from the filename for sorting.
        E.g., 'circuit_14.qasm' -> 14. If none found, return +∞ so that
        such files sort to the end.

        :param filename: Filename string, e.g. 'circuit_14.qasm'.
        :return: An integer suffix if found, else float('inf').
        """
        try:
            base = filename.replace('.qasm', '')
            parts = base.split("_")[::-1]
            for part in parts:
                try:
                    return int(part)
                except ValueError:
                    continue
            return float('inf')
        except Exception:
            return float('inf')

    def modify_result_folder(self, new_folder: str):
        """
        Change the results folder if the given path does not already exist.
        Otherwise, print a warning.

        :param new_folder: The path to the new results folder.
        """
        if not os.path.exists(new_folder):
            self.results_folder = new_folder
            os.makedirs(self.results_folder)
        else:
            print(f"Folder already exists: {new_folder}. Try using a different path.")

    def process_all_files(self, file_indices=None):
        """
        Process either all QASM files or a selected subset. Results are aggregated
        in a single Excel workbook.

        :param file_indices: A list of indices specifying which files to process.
        If None, process all.
        """
        # Prepare sub-folders
        result_subfolder = os.path.join(self.results_folder, f"Rb{self.interaction_radius:.3g}Re{self.extended_radius:.3g}")
        embeddings_subfolder = os.path.join(result_subfolder, "embeddings")
        partitions_subfolder = os.path.join(result_subfolder, "partitions")
        os.makedirs(embeddings_subfolder, exist_ok=True)
        os.makedirs(partitions_subfolder, exist_ok=True)

        # Create a master Excel workbook for the entire benchmark
        self.master_workbook = Workbook()
        self.master_sheet = self.master_workbook.active
        self.master_sheet.append([
            'QASM File',
            'Num Qubits',
            'Num CZ Gates',
            'Circuit Depth',
            'Fidelity',
            'Movement Fidelity',
            'Num Movement Ops',
            'Num Transferred Qubits',
            'Num Moves',
            'Total Move Distance',
            'Num Gate Cycles',
            'Num Partitions',
            'Elapsed Time (s)',
            'Total_T (from fidelity calc)',
            'Idle Time'
        ])

        # If no indices specified, process all files
        if file_indices is None:
            file_indices = range(len(self.qasm_files))

        # Process each specified file
        for idx in file_indices:
            qasm_file = self.qasm_files[idx]
            print(f"Processing: {qasm_file}")
            if self.if_verify:
                print(f"Also verify the result of {qasm_file}")

            processor = SingleFileProcessor(
                qasm_filename=qasm_file,
                circuit_folder=self.circuit_folder,
                benchmark_name=self.benchmark_name,
                interaction_radius=self.interaction_radius,
                extended_radius=self.extended_radius,
                result_path=result_subfolder,
                embeddings_path=embeddings_subfolder,
                partitions_path=partitions_subfolder,
                read_embeddings=self.read_embeddings,
                save_partitions_and_embeddings=self.save_partitions_and_embeddings,
                save_circuit_results=self.save_circuit_results,
                save_benchmark_results=self.save_benchmark_results,
                if_verify= self.if_verify
            )

            # Returns one row of aggregated stats
            row_data = processor.process_qasm_file()
            self.master_sheet.append(row_data)

        # Optionally append global parameters at the bottom
        params_dict = set_parameters(True)
        param_log_row = []
        for key, val in params_dict.items():
            param_log_row.append(str(key))
            param_log_row.append(str(val))
        self.master_sheet.append(param_log_row)

        # Save the aggregated results if requested
        if self.save_benchmark_results:
            master_file_path = os.path.join(result_subfolder, f'{self.benchmark_name}_summary.xlsx')
            self.master_workbook.save(master_file_path)


# ------------------------------------------------------------------------------
# Script entry point for command line usage
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize and run the DasAtom benchmark processor.")

    parser.add_argument("benchmark_name", type=str, help="Name of the benchmark.")
    parser.add_argument("circuit_folder", type=str, help="Path to the folder containing .qasm files.")
    parser.add_argument("--interaction_radius", type=int, default= 2, help="Interaction radius (default=2).")
    parser.add_argument("--results_folder", type=str, help="Folder where results are stored (default: res/{benchmark_name}).")
    parser.add_argument("--read_embeddings", action="store_true", default=False, help="Read precomputed embeddings/partitions.")
    parser.add_argument("--padused", type=bool, default=False, help="Whether to use a specialized embedding tool (not used in code).")
    parser.add_argument("--save_embeddings", action="store_true", default=True, help="Save partition/embedding JSONs (default=True).")
    parser.add_argument("--no_save_embeddings", action="store_false", dest="save_embeddings", help="Do not save partitions/embeddings.")
    parser.add_argument("--save_circuit_results", action="store_true", default=True, help="Save circuit-level XLSX logs (default=True).")
    parser.add_argument("--no_save_circuit_results", action="store_false", dest="save_circuit_results", help="Do not save circuit-level logs.")
    parser.add_argument("--save_benchmark_results", action="store_true", default=True, help="Save summary XLSX at benchmark-level (default=True).")
    parser.add_argument("--no_save_benchmark_results", action="store_false", dest="save_benchmark_results", help="Do not save summary XLSX.")
    parser.add_argument("--verify", type=bool, default=True, help="Whether to verify the whole process.")

    args = parser.parse_args()

    das_atom = DasAtom(
        benchmark_name=args.benchmark_name,
        circuit_folder=args.circuit_folder,
        interaction_radius=args.interaction_radius,
        results_folder=args.results_folder,
        read_embeddings=args.read_embeddings,
        save_partitions_and_embeddings=args.save_embeddings,
        save_circuit_results=args.save_circuit_results,
        save_benchmark_results=args.save_benchmark_results,
        if_verify = args.verify
    )
    das_atom.process_all_files()