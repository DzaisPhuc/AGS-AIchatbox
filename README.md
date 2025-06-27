# AGS AI Agent

A multi-agent AI system for agricultural product sales and customer requirement processing, powered by [CrewAI](https://crewai.com). This project uses intelligent agents to handle customer inquiries, match product specifications, and generate comprehensive sales proposals for agricultural products.

## Overview

The AGS AI Agent system consists of four specialized AI agents that work together to:
- Collect and clarify customer requirements through interactive conversations
- Plan information flow and identify constraints
- Match customer needs with available products
- Generate professional sales proposals

## Architecture

The system uses a multi-agent architecture with the following components:

### Agents
- **Customer Requirements Agent** ([`receiver`](src/ags_ai_agent/config/agents.yaml)): Gathers detailed product requirements from customers
- **Sales Planner** ([`manager`](src/ags_ai_agent/config/agents.yaml)): Interprets requirements and plans information flow
- **Product Match Agent** ([`analyzer`](src/ags_ai_agent/config/agents.yaml)): Matches customer requirements with supplier capabilities
- **Proposal Generator Agent** ([`compiler`](src/ags_ai_agent/config/agents.yaml)): Compiles findings into comprehensive proposals

### Tools
- **Milvus Knowledge Search** ([`MyCustomTool`](src/ags_ai_agent/tools/custom_tool.py)): Semantic search over embedded documents
- **Interactive Input Tool**: Collects information from users during conversations

### Knowledge Base
The system uses Milvus vector database for semantic search across:
- Product specifications (PDF documents)
- User preferences ([`user_preference.txt`](knowledge/user_preference.txt))
- FAQ and documentation

## Installation

### Prerequisites
- Python ≥3.10 <3.13
- Docker and Docker Compose (for Milvus)
- OpenAI API key

### Setup

1. **Install UV package manager** (if not already installed):
```bash
pip install uv
```

2. **Clone and navigate to the project**:
```bash
cd ags_ai_agent
```

3. **Install dependencies**:
```bash
crewai install
```
or
```bash
uv sync
```

4. **Set up environment variables**:
Create a `.env` file in the root directory and add your OpenAI API key:
```bash
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Start Milvus vector database**:
```bash
docker-compose up -d
```

6. **Ingest knowledge documents**:
```bash
python -c "from src.ags_ai_agent.tools.vector_db import ingest_documents; ingest_documents()"
```

## Usage

### Running the System

To start the AI agent crew and begin processing customer requirements:

```bash
crewai run
```

This will:
1. Initialize all four AI agents
2. Start an interactive conversation to collect customer requirements
3. Process the requirements through the agent pipeline
4. Generate a comprehensive proposal in [`report.md`](report.md)

### Example Workflow

1. **Customer Interaction**: The system will ask for product specifications, certifications, volumes, etc.
2. **Requirement Processing**: Agents analyze and match requirements with available products
3. **Proposal Generation**: A detailed proposal is created covering:
   - Product Specifications
   - Certifications
   - Shelf Life
   - INCOTERM details
   - Volume requirements
   - Packaging specifications
   - Payment terms
   - Final pricing
   - Any deviations from original requests

## Configuration

### Agent Configuration
Modify agent behaviors in [`src/ags_ai_agent/config/agents.yaml`](src/ags_ai_agent/config/agents.yaml):
- Roles and goals
- Backstories and personalities
- Delegation permissions

### Task Configuration
Customize task workflows in [`src/ags_ai_agent/config/tasks.yaml`](src/ags_ai_agent/config/tasks.yaml):
- Task descriptions and requirements
- Expected outputs
- Agent assignments

### Knowledge Base
Add documents to the [`knowledge/`](knowledge/) folder:
- PDF product specifications
- Text files with company information
- FAQ documents

## Development

### Project Structure
```
ags_ai_agent/
├── src/ags_ai_agent/
│   ├── config/           # Agent and task configurations
│   ├── tools/           # Custom tools and vector database
│   ├── crew.py          # Main crew definition
│   └── main.py          # Entry point
├── knowledge/           # Knowledge base documents
├── docker-compose.yml   # Milvus database setup
└── README.md           # This file
```

### Key Files
- [`crew.py`](src/ags_ai_agent/crew.py): Defines agents, tasks, and crew configuration
- [`custom_tool.py`](src/ags_ai_agent/tools/custom_tool.py): Milvus search tool implementation
- [`vector_db.py`](src/ags_ai_agent/tools/vector_db.py): Document ingestion and embedding functions
- [`docker-compose.yml`](docker-compose.yml): Milvus vector database setup

### Adding New Knowledge
1. Place documents in the [`knowledge/`](knowledge/) folder
2. Run the ingestion script:
```python
from src.ags_ai_agent.tools.vector_db import ingest_documents
ingest_documents()
```

### Training and Testing
```bash
# Train the crew
python main.py train <iterations> <filename>

# Test the crew
python main.py test <iterations> <model_name>

# Replay specific tasks
python main.py replay <task_id>
```

## Technologies Used

- **CrewAI**: Multi-agent framework
- **OpenAI GPT-4**: Language models for agents
- **Milvus**: Vector database for semantic search
- **Docker**: Containerization for database services
- **Python**: Core development language

## Support

For support, questions, or feedback:
- Visit [CrewAI documentation](https://docs.crewai.com)
- Check the [CrewAI GitHub repository](https://github.com/joaomdmoura/crewai)
- [Join the CrewAI Discord](https://discord.com/invite/X4JWnZnxPb)

## License

This project is built using the CrewAI framework. Please refer to individual component licenses for specific terms.