# ReAgentAI

### Setup
requires-python = ">=3.10"

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
download_public_data data
```