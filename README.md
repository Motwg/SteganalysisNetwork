# Requirements
- Installed Python 3.8
- [Optional] Installed Nvidia CUDA 10.1 https://developer.nvidia.com/cuda-downloads
  (improve performance, requires cuDNN)
- [Optional] Installed cuDNN 7.6 https://developer.nvidia.com/cudnn

# Virtual environment configuration (optional)
- Proceed to the catalog for the project
- Run `python3 -m venv stego-venv`
- On **Unix** or **MacOS**, run: `stego-venv/bin/activate`
- On **Windows**, run: `stego-venv\Scripts\activate.bat`

# Install libraries
- Proceed to the catalog for the project
- Run `python3 pip install -r requirements.txt`
  

In order to train network run `python3 train_model.py` in project catalog.
For evaluation use `python3 test_model.py` script.