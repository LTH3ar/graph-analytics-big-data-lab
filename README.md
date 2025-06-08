# graph-analytics-big-data-lab
## How to run the code
### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook or JupyterLab or any Python IDE
- Required Python packages installed (see `requirements.txt`)
### Steps to run
1. Clone the repository:
2. Navigate to the cloned directory:
   ```bash
   cd graph-analytics-big-data-lab
   ```
3. Install the required packages and download the datasets:
   ```bash
   pip install -r requirements.txt
   chmod +x kaggle.sh
    ./kaggle.sh
   ```
4. Open Jupyter Notebook to use like normal
5. Alternatively, you can run the Python scripts directly in your IDE or terminal.
   ```bash
   python main.py <model_name: gcn/gat/stgat> <horizon_number: int> <batch_size: int> 
    ```
