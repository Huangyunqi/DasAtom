# Usage Guide

This document provides instructions on how to create the Python environment for this project and lists the main packages used.

## Creating the Environment

1. **Clone the Repository**

   First, clone the repository to your local machine:

   ```bash
   git clone git@github.com:Huangyunqi/NAQCT.git
   cd NAQCT
   ```

2. **Set Up a Virtual Environment**

   It is recommended to use a virtual environment to manage dependencies. You can create one using `venv`:

   ```bash
   python -m venv .venv
   ```

   Activate the virtual environment:

   - On Windows:

     ```bash
     .\.venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source .venv/bin/activate
     ```

3. **Install the Dependencies**

   Use `pip` to install the required packages listed in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## Main Packages Used

The following are the main packages used in this project:

- **rustworkx**:
- **networkx** :
- **qiskit** :

## Additional Notes

- Ensure that you have Python 3.6 or higher installed on your system.
- If you encounter any issues during the installation, ensure that all required system dependencies are installed.

## Running the Project

After setting up the environment and installing the dependencies, you can run the project scripts as follows:

<!-- ```bash
python main.py
``` -->


## Deactivating the Environment

Once you are done working with the project, you can deactivate the virtual environment:

```bash
deactivate
```

That's it! You have successfully set up the Python environment for this project. If you have any questions or issues, please refer to the project documentation or contact the project maintainers.
