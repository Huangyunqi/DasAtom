# Benchmark Folder

This folder contains a collection of benchmarks used in This project. These benchmarks are sourced from various trusted repositories and have been adapted for use in our project.

## Folder Sources

Below is a breakdown of the folders within this directory, along with the original sources of the benchmarks:

- **mqt-bench/**
  Contains benchmarks downloaded from the [MQT Benchmark Library](https://www.cda.cit.tum.de/mqtbench/#:~:text=The%20MQT%20Benchmark%20Library%20(MQT%20Bench)%20provides%20a%20single%20benchmark). Includes benchmarks `DJ`, `GHZ`, `local random`, and `Wstate`. The original two-qubit gate `CX` has been transpiled to `CZ` using `qiskit`.

- **3-regular-graph/**
  Contains benchmarks generated from the [Enola repository](https://github.com/UCLA-VAST/Enola). These benchmarks involve 3-regular graphs.

- **Q_Tetris/**
  Contains benchmarks collected from the [Q-Tetris repository](https://github.com/s4plus/Q-Tetris).

- **qiskit-bench/**
  Contains benchmarks generated from the [Qiskit circuit library](https://docs.quantum.ibm.com/api/qiskit/circuit_library). Includes `QFT` and `Quantum Volume` benchmarks. The original two-qubit gate `CX` has been transpiled to `CZ` using `qiskit`.

## Acknowledgements

We would like to acknowledge and thank the original authors for providing these valuable resources.

If you are the original author and have any concerns or requests regarding the use of your benchmarks, please feel free to contact us.
