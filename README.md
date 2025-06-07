# ReAgentAI

### Setup
Supported Python versions: 3.10 and 3.11.

```sh
uv sync && source .venv/bin/activate
```
or
```sh
python -m venv .venv --upgrade-deps &&
source .venv/bin/activate &&
pip install . &&
echo "Setup completed successfully"
```

Create a `.env` file in the root directory with the following content:
```env
GEMINI_API_KEY=your_gemini_api_key
```

You need a trained model and a stock collection. You can download a public available model based on USPTO and a stock
collection from ZINC database.
```sh
mkdir data
download_public_data data
```

### Troubleshooting
Problem:
```
ImportError: PATH_TO_PROJECT_DIRECTORY/.venv/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so: cannot enable executable stack as shared object requires: Invalid argument
```
Possible solution:
```
execstack -c PATH_TO_PROJECT_DIRECTORY/.venv/lib/python3.10/site-packages/onnxruntime/capi/onnxruntime_pybind11_state.cpython-310-x86_64-linux-gnu.so
```
