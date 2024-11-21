
# Usage Guide
## About This Repository
This repository accompanies the research paper [**"DasAtom: A Divide-and-Shuttle Atom Approach to Quantum Circuit Transformation"**](https://arxiv.org/abs/2409.03185), detailing the methods of our work.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone git@github.com:Huangyunqi/NAQCT.git
   cd NAQCT
   ```

2. **Create and Activate Virtual Environment:**

   ```bash
   python -m venv .venv
   ```

   - Windows: `.\.venv\Scripts\activate`
   - macOS/Linux: `source .venv/bin/activate`

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Main Packages Project used

- **rustworkx**: Use for check subgraph isomorphism and find embeddings.
- **qiskit**: Handles QASM files and gate optimizations.

## Project Folder Structure

- **`DasAtom.py`**: The main Python program of this project.
- **`DasAtom_fun.py`**: Contains supporting functions used by `DasAtom.py`.
- **`Enola/`**: Responsible for generating and visualizing movement sequences. For more details, refer to the [Enola folder README](Enola/README.md).
- **`Data/`**: Contains the benchmark datasets used in this project. See the [Data folder README](Data/README.md) for further information.


## Running the Project example

Use `make` to run project scripts in the background:

```bash
make
```

The `Makefile` automates two tasks:

- Executes the `qft` script, logging the output to `qft.log`, which processes benchmarks from `qft_5.qasm` to `qft_20.qasm`.
- Executes the `tetris` script, logging the output to `tetris.log`, which processes the benchmarks from `Q_Tetris`.


If you have any questions or issues, please refer to us.
