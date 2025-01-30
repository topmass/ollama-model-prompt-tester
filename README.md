# Ollama API UI

A Textual-based terminal user interface for interacting with the Ollama API, focusing on prompt variance testing.

## Features

- List available Ollama models
- Pull models from Ollama
- Configure system and user prompts
- Set model parameters (temperature, max tokens, context window)
- Run concurrent prompts to test response variance
- Export results to CSV (coming soon)

## Requirements

- Python 3.8 or higher
- Ollama installed and running locally (https://ollama.ai)

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd ollama-api-ui
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Make sure Ollama is running locally (default: http://localhost:11434)

2. Run the UI:
```bash
python ollama_ui.py
```

## Keyboard Shortcuts

- `d`: Toggle dark mode
- `q`: Quit the application

## UI Elements

- **Model Name**: Enter the name of the Ollama model to use
- **System Prompt**: Optional system prompt for the model
- **User Prompt**: The main prompt to send to the model
- **Parameters**:
  - Temperature: Controls randomness (0.0 to 1.0)
  - Max Tokens: Maximum number of tokens to generate
  - Context Window: Size of the context window
  - Concurrency Count: Number of times to run the same prompt
- **Buttons**:
  - List Models: Show available local models
  - Pull Model: Download a model from Ollama
  - Run: Execute the prompts
  - Export CSV: Save results to CSV (coming soon)

## License

MIT 