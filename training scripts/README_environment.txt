To reproduce the computational environment required for reproduce the results in LEADS, we provide an leads_environment.yml file that specifies all necessary dependencies along with their exact versions. The following steps outline how to recreate this environment using Conda.

Prerequisites. Ensure that either Anaconda or Miniconda is installed on your system. If not, please refer to the official Conda documentation for installation instructions appropriate to your operating system.

Environment setup. Execute the command below to construct an identical Conda environment from the provided specification:

conda env create -f leads_environment.yml

This will automatically resolve and install all required packages, including those managed by both the conda and pip package managers, as declared in the file.

Activation. Once the installation completes, activate the newly created environment with:

conda activate *environment-name*

The environment name is defined inside the leads_environment.yml file (look for the name field). If you prefer a custom name, you can override it by appending -n *your-name* to the conda env create command.

Verification. To confirm that the environment has been correctly reproduced, you may run a quick sanity check, such as importing the core libraries (e.g., python -c "import keras; import tensorflow; print('Environment ready')"). All subsequent experiments and scripts should be executed within this activated environment to guarantee consistent behavior.

Important notes: While the leads_environment.yml file ensures software-level reproducibility, certain hardware-specific dependencies (most notably the CUDA toolkit and GPU drivers) must be installed separately and matched to the capabilities of your computing infrastructure. The code was developed and tested with CUDA 11.x; adjustments may be necessary if your system differs significantly. Please consult the official CUDA compatibility guides if you encounter runtime errors related to GPU support.

Additional note: For training based on Reformer, you need to first deploy Reformer locally, and then use reformer_environment.yml file (rather than leads_environment.yml file). Please follow the instructions on the HuggingFace website.