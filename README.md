#  ESDM Recommendation Engine

[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app/)  
![GitHub repo size](https://img.shields.io/github/repo-size/rajith23132/ESDM-Recommendation-Engine)  
![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)  

> **One engine. Every stakeholder. Better choices**  
> AI-powered recommendation system for Electronic System Design & Manufacturing (ESDM) components with **sustainability insights** and **technical benchmarking**

##  Key Features
| Feature | Description |
|---------|-------------|
| **🔍 Smart Component Finder** | NLP-powered search with Gemini AI |
| **👁️ Visual Search** | Image-based component identification |
| **📊 Comparative Analysis** | Interactive radar charts & specs comparison |
| **♻️ Green Metrics** | Environmental impact scoring |

##  Tech Stack
- **Frontend**: Streamlit, Plotly  
- **Backend**: ChromaDB, Sentence Transformers  
- **AI/ML**: Gemini 2.0 Flash, LangChain  

##  Data Pipeline
### `build_embeddings.py`
```python
# Processes ESDM_Dataset.xlsx to:
1. Generate vector embeddings
2. Create ChromaDB collection
3. Store metadata + JSON documents
```

##  Quick Start
```bash
# Clone & setup
git clone https://github.com/rajith23132/ESDM-Recommendation-Engine.git
cd ESDM-Recommendation-Engine

# Install dependencies
pip install -r requirements.txt

# Build embeddings
python build_embeddings.py

# Launch app
streamlit run app.py
```

##  Project Structure
```
ESDM-Recommendation-Engine/
├── app.py            # Main application
├── build_embeddings.py  # Data pipeline
├── chroma_db/        # Vector database
├── requirements.txt
└── README.md
```

##  How to Contribute
1. Fork the repository  
2. Create a feature branch  
3. Submit a Pull Request  


---

> **Capstone Project 2024-25**  
> Indian Institute of Information Technology, Delhi
