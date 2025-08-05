# RAG System - Agricultural Q&A

A comprehensive Retrieval-Augmented Generation (RAG) system for agricultural question answering, featuring multiple expert approaches and advanced techniques.

## Features

### Expert Modes

1. **AllExpert** - Uses entire document database for maximum accuracy
2. **PartialExpert** - Uses top-K snippets for speed and partial answers
3. **SqlExpert** - Text-to-SQL conversion for structured data queries
4. **RawLlmExpert** - Direct LLM responses without retrieval
5. **AdaptiveExpert** - Self-RAG with adaptive gating
6. **SelfAskExpert** - Self-Ask approach with sufficiency checking

### Key Improvements

- **Type Safety**: Full type hints throughout the codebase
- **Error Handling**: Comprehensive exception handling with graceful fallbacks
- **Configuration Management**: Centralized configuration with YAML support
- **Logging**: Structured logging with performance monitoring
- **Modular Design**: Clean separation of concerns with abstract base classes
- **Performance Monitoring**: Built-in timing and metrics collection

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables (optional):
```bash
export MODEL_DIR="./selfrag_llama2_7b"
```

## Configuration

The system uses `config.yaml` for configuration. Key settings include:

```yaml
# Model Settings
models:
  default: "Mistral-nemo"
  sql_coder: "sqlcoder:15b"
  selfrag: "./selfrag_llama2_7b"

# Model Parameters
model_params:
  temperature: 0.0
  top_p: 1.0
  top_k: 40

# Retrieval Settings
retrieval:
  default_ndocs: 3
  max_iterations: 3
  threshold: 0.5
```

## Usage

### Interactive Mode

Run the main script for interactive Q&A:

```bash
python rag.py
```

Available modes:
- `all` - Full database RAG (accuracy-focused)
- `partial` - Top-K snippets RAG (speed-focused)
- `sql` - Text-to-SQL queries
- `raw_llm` - Direct LLM responses
- `adaptive` - Self-RAG with gating
- `self_ask` - Self-Ask approach

### Programmatic Usage

```python
from rag import create_expert_instances
from src.store_db import load_stores

# Initialize retrievers
retr_qna, retr_crop, retr_soil, qna_sql_path, crop_sql_path = load_stores(ndocs=3)
retriever_map = {"qna": retr_qna, "crop": retr_crop, "soil": retr_soil}

# Create expert instances
experts = create_expert_instances(retriever_map, "qna", qna_sql_path, crop_sql_path)

# Use an expert
expert = experts["partial"]
answer = expert.handle("What crops grow well in nitrogen-rich soil?")
print(answer)
```

## Architecture

### Core Components

1. **BaseExpert** - Abstract base class for all experts
2. **Configuration System** - Centralized config management
3. **Logging System** - Structured logging with performance metrics
4. **Prompt Management** - Centralized prompt templates
5. **Database Integration** - SQLite and vector store support

### Expert Implementations

#### AllExpert
- Uses entire document database
- Maximum accuracy, slower performance
- Best for complex queries requiring comprehensive information

#### PartialExpert
- Uses top-K retrieved snippets
- Balanced speed and accuracy
- Good for most general queries

#### SqlExpert
- Converts natural language to SQL
- Executes queries on structured data
- Ideal for data-driven questions

#### AdaptiveExpert
- Implements Self-RAG approach
- Adaptive gating for retrieval decisions
- Advanced reasoning capabilities

#### SelfAskExpert
- Self-Ask methodology
- Sufficiency checking
- Iterative question decomposition

## Performance Monitoring

The system includes built-in performance monitoring:

```python
from src.logger import setup_logging

logger = setup_logging(level="INFO")
logger.log_expert_call("PartialExpert", "What is crop rotation?", 2.5, True)
logger.log_retrieval("qna_retriever", "crop rotation", 5, 0.8)
```

## Error Handling

All experts include comprehensive error handling:

- Model loading failures
- Database connection issues
- SQL execution errors
- Retrieval failures
- Network timeouts

## Development

### Adding New Experts

1. Inherit from `BaseExpert`
2. Implement `setup()` and `handle()` methods
3. Add to expert factory in `create_expert_instances()`

```python
class NewExpert(BaseExpert):
    def setup(self, retriever_mode: str) -> None:
        # Initialize components
        pass
    
    def handle(self, question: str) -> str:
        # Implement expert logic
        return "answer"
```

### Configuration Management

Use the configuration system for new settings:

```python
from src.config import get_config

config = get_config()
model_config = config.get_model_config("default")
```

## Testing

Run tests with:

```bash
python -m pytest tests/
```

## Logging

Logs are stored in the `logs/` directory with timestamps:

```
logs/
├── rag_system_20241201_143022.log
├── rag_system_20241201_150145.log
└── ...
```

## Troubleshooting

### Common Issues

1. **Model Loading Failures**
   - Check model paths in config
   - Ensure Ollama is running
   - Verify model names

2. **Database Errors**
   - Check database file paths
   - Verify SQLite installation
   - Ensure proper permissions

3. **Memory Issues**
   - Reduce `gpu_memory_utilization` in vLLM config
   - Lower `max_num_seqs` for vLLM
   - Use smaller models

### Debug Mode

Enable debug logging:

```python
logger = setup_logging(level="DEBUG")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Add your license information here]

## Acknowledgments

- Self-RAG implementation based on [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)
- LangChain for RAG framework
- Ollama for local model serving 