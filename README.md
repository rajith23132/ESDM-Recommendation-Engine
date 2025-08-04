
# ESDM Recommendation Engine

![Banner](banner-8192025_1280.png)

An AI-powered recommendation system for sustainable electronic components, helping engineers, procurement teams, and designers make environmentally-conscious choices in the Electronics System Design & Manufacturing (ESDM) sector.

##  Features

- **Smart Component Finder**: Natural language search for electronic components
- **Multi-dimensional Comparison**: Compare products across technical, economic, and sustainability metrics
- **Manufacturer Analysis**: Evaluate companies based on their product portfolios
- **Visual Search**: Upload component images for identification and matching
- **Gemini AI Integration**: Advanced LLM-powered analysis and recommendations

##  Tech Stack

- **Backend**: Python, ChromaDB (vector database)
- **AI Models**: 
  - Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings
  - Google Gemini 2.0 Flash for natural language processing
- **Frontend**: Streamlit (interactive web interface)
- **Data**: Excel/CSV based product catalog with sustainability metrics

##  Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/esdm-recommender.git
   cd esdm-recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your Google API key:
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   ```

   Or add it to secrets.toml for Streamlit:
   ```toml
   # .streamlit/secrets.toml
   GOOGLE_API_KEY = "your-api-key"
   ```

##  Usage

1. Prepare your dataset:

   - Place your Excel file (`ESDM_Dataset.xlsx`) in the project root
   - Format should match the sample structure (see `sample_data/`)

2. Generate embeddings:
   ```bash
   python build_embeddings.py
   ```

3. Run the application:
   ```bash
   streamlit run app13.py
   ```

##  Project Structure

```
esdm-recommender/
├── app13.py                # Main Streamlit application
├── build_embeddings.py     # Embedding generation script
├── gemini_engine.py        # Gemini AI integration
├── requirements.txt        # Python dependencies
├── chroma_db/              # Vector database storage
├── feedback/               # User feedback storage
├── sample_data/            # Example datasets
└── images/                 # Application images
```

##  Application Tabs

- **Recommendation Engine**: Natural language search for components
- **Compare Products**: Side-by-side technical and sustainability comparison
- **Product Profiler**: Detailed component analysis
- **Manufacturers**: Company directory and evaluation
- **Visual Search**: Image-based component identification

##  Data Requirements

Your dataset should include these key fields (see `sample_data/schema.md` for details):

- Product identifiers (Name, ID, Type)
- Technical specifications
- Sustainability metrics (Green Index, Recyclability Score)
- Manufacturer information
- Pricing and availability

##  Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

##  Contact

For questions or support:

- Rajith Ramachandran - rajith.ramachandran99@gmail.com
- Abhishek Tibrewal - abhishekasd63@gmail.com

<div align="center"> <i>Built with ❤️ for sustainable electronics</i> </div>
