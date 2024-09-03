# External Code Attribution

This folder contains code sourced from an external GitHub repository, specifically designed to enhance the functionality of our project. While this code is not central to the core operations of our application, we have made minor modifications to ensure seamless integration.

## Source Repository
The original code can be found at:
[Enola](https://github.com/UCLA-VAST/Enola)

## Organization
This project was originally developed by the [UCLA VAST Lab](https://vast.cs.ucla.edu/), which has generously made it available to the open-source community. We are immensely grateful for their contributions and are pleased to incorporate their work into our project.

## License
The original code is distributed under the BSD 3-Clause License, which permits use, modification, and distribution. For more details, please see the full license text [here](https://opensource.org/licenses/BSD-3-Clause).

## Acknowledgements
We extend our special thanks to the [UCLA VAST Lab](https://vast.cs.ucla.edu/) for their dedication and support to the community. Their innovative work has significantly enhanced the capabilities of numerous projects, including ours.

## Modifications
The `route.py` file is used to generate the movement sequences. We streamlined the code by removing unnecessary functions and retaining only `compatible_2D`, `maximalis_solve_sort`, and `maximalis_solve`. Additionally, we introduced a new class, `QuantumRouter`, to implement our methods more effectively.

Unlike the Enola setup, our implementation does not involve dual movements, which reduces the number of possible movements. Consequently, we did not use movement windows.

By integrating `route.py` with `codegen.py` and `animation.py`, we can visualize the movement sequences.

## Contact
For more information on the original code or our modifications, please contact us at [by.gdc@outlook.com].
