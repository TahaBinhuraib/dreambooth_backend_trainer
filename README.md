# Dreambooth trainer backend
## Uploads trained models to AWS S3.

## Installation and Setup
1. Create your anaconda env
   ``` bash
   conda create -n myenv python=3.8
   ```

2. Install packages using [pip](https://pypi.org/project/pip/)
    ```bash
    pip install -r requirements.txt
    ```
3. Download base v1_4 SD model
   ```bash
   python download_model.py
   ```
4. Run the code
   ``` bash
   uvicorn main:app --reload
   ```
