# === ESDM GreenTech Recommender ===
import streamlit as st
import chromadb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
import re
import requests
from typing import Dict, List, Optional
from datetime import datetime
from PIL import Image
import plotly.express as px 

# Add custom CSS to center the tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
    }
    
    .stTabs [data-baseweb="tab"] {
        margin: 0 10px;
    }
</style>
""", unsafe_allow_html=True)

# === Setup ===
st.set_page_config(page_title="ESDM Recommendation Engine", layout="wide")
st.markdown("""
    <h1 style='text-align: center;'>🔰 ESDM Recommendation Engine 🔰</h1>
    <p style='text-align: center; color: #666;'>One engine. Every stakeholder. Better choices</p>
    """, unsafe_allow_html=True)

# Center-aligned image with proper width control
col1, col2, col3 = st.columns([1, 2, 1])  # Wider middle column for image

with col2:  # This centers the image
    try:
        header_img = Image.open("banner-8192025_1280.png")
        st.image(
            header_img,
            width=700,  # Fixed width
            use_container_width=False,  # Disable auto column width
            output_format="auto",  # Maintains original format
        )
    except FileNotFoundError:
        st.warning("Header image not found. Using placeholder.")
        st.image(
            "Fokus-AI-Phonlamai-Shutterstock.jpg",
            width=500,  # Same width as main image
            use_column_width=False,
            caption="Default Banner"
        )

# Optional: Add CSS for better image presentation
st.markdown("""
<style>
    .stImage > img {
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    .stImage > div {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# === Gemini Setup ===
@st.cache_resource
def load_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        google_api_key="AIzaSyBgNyOek4g4RhmY_-JG_lxG1n5kA-yKUE8"  
    )

llm = load_llm()

# === Load Vector DB ===
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection("esdm_products")

model = load_embedding_model()
collection = load_chroma_collection()

# === Utility Functions ===
def extract_keywords_with_gemini(query: str) -> Dict[str, str]:
    """Enhanced keyword extraction for Gemini 2.0 Flash"""
    if not query.strip():
        return {"product_type": query, "features": "", "use_case": ""}

    # Get API key (prefer secrets, fallback to env var)
    api_key = (
        st.secrets.get("GOOGLE_API_KEY") 
        or os.getenv("GOOGLE_API_KEY") 
        or "your-api-key"  # fallback for testing
    )
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    prompt = f"""
    As an electronics component expert, extract these 3 elements from the query:
    
    Query: "{query}"
    
    Return ONLY this JSON format:
    {{
      "product_type": "<specific component type>",
      "features": "<key technical requirements>",
      "use_case": "<primary application>"
    }}
    
    Examples:
    - "I need a low power microcontroller for IoT sensors" → 
      {{
        "product_type": "microcontroller",
        "features": "low power consumption",
        "use_case": "IoT sensors"
      }}
    - "Show me durable connectors for automotive use" → 
      {{
        "product_type": "electrical connectors",
        "features": "durable, vibration resistant",
        "use_case": "automotive"
      }}
    """

    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 200
        }
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()  # Raises exception for 4XX/5XX errors
        
        # Parse the response
        result = response.json()
        if not result.get("candidates"):
            raise ValueError("No candidates in response")
            
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
        cleaned_text = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
        
        return json.loads(cleaned_text)
        
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        # Fallback to returning the query as product_type
        return {"product_type": query, "features": "", "use_case": ""}

def retrieve_similar_products(extracted_keywords: Dict[str, str], top_k: int = 5) -> List[Dict]:
    """Enhanced retrieval with fallback to semantic search"""
    # Use raw query if extraction failed
    if not any(extracted_keywords.values()):
        query = extracted_keywords.get("product_type", "")
    else:
        query = " ".join([
            extracted_keywords.get("product_type", ""),
            extracted_keywords.get("features", ""),
            extracted_keywords.get("use_case", "")
        ]).strip()

    if not query:
        return []

    try:
        query_embedding = model.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents"]
        )

        products = []
        for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
            if meta:
                product = {
                    "product_id": meta.get("Product_id"),
                    "product_name": meta.get("Product Name"),
                    "product_type": meta.get("Product Type"),
                    "company_name": meta.get("Company Name"),
                    "specs": doc,
                    "green_index": meta.get("Green Index"),
                    "innovation_score": meta.get("Innovation Score"),
                    "recyclability_score": meta.get("Recyclability Score"),
                    "repairability_index": meta.get("Repairability Index"),
                    "datasheet_score": meta.get("Datasheet Score"),
                    "product_url": meta.get("Product URL"),
                    "lead_time_days": meta.get("Lead Time (Days)"),
                    "carbon_footprint": meta.get("Carbon Footprint (g)"),
                    "esg_certified": meta.get("ESG Certified"),
                    "support_community": meta.get("Support Community"),
                    "use_case": meta.get("Use Case"),
                    "lifecycle_stage": meta.get("Lifecycle Stage"),
                    "temperature_range": meta.get("Temperature Range (°C)"),
                    "compliance_standards": meta.get("Compliance Standards"),
                    "warranty": meta.get("Warranty (Years)"),
                    "hazardous_materials": meta.get("Hazardous Materials"),
                    "voltage_range": meta.get("Voltage Range (V)"),
                    "pin_count": meta.get("Pin Count"),
                    "mtbf_hours": meta.get("MTBF (Hours)"),
                    "ai_ready": meta.get("AI Ready"),
                    "manufacturing_capacity": meta.get("Manufacturing Capacity (Units/Year)"),
                    "delivery_rate": meta.get("On-Time Delivery Rate (%)"),
                    "company_url": meta.get("company_url"),
                    "price": meta.get("Price (INR)"),
                    "power_consumption": meta.get("Power Consumption (mW)"),
                    "packaging_type": meta.get("Packaging Type"),
                    "supply_chain_risk_score": meta.get("Supply Chain Risk Score"),
                    "description": meta.get("Description")
                }
                products.append(product)
        return products

    except Exception as e:
        st.error(f"Search failed: {str(e)}")
        return []

def generate_response(products: List[Dict], extracted_features: str = "") -> str:
    """Generate LLM explanation for recommendations"""
    if not products:
        return "No products found matching your criteria."

    context = "\n".join([
        f"- {p['product_name']} ({p['product_type']}): {p.get('description', '')[:100]}... | "
        f"Price: ₹{p.get('price', 'N/A')} | "
        f"Green Index: {p.get('green_index', 'N/A')}, "
        f"Innovation: {p.get('innovation_score', 'N/A')}, "
        f"Power: {p.get('power_consumption', 'N/A')}mW | "
        f"Use: {p.get('use_case', 'N/A')}" 
        for p in products
    ])

    # Add feature matching information if features were provided
    feature_analysis = ""
    if extracted_features:
        feature_analysis = f"\n\nRequested Features: {extracted_features}\n" + \
                         "\n".join([
                             f"- {p['product_name']}: {'✅' if extracted_features.lower() in str(p.get('specs', '')).lower() else '❌'} Matches features"
                             for p in products
                         ])

    prompt = f"""
As an ESDM industry expert, analyze these components:
{context}
{feature_analysis}

Recommend top 5 most suitable options considering:
1. Technical match to requested features: {extracted_features if extracted_features else 'None specified'}
2. Price-to-performance ratio
3. Sustainability (green index)
4. Specifications (especially matching: {extracted_features})
5. Description
6. Application fit

Format your response with:
- 🏆 Top Recommendations (ranked by feature matching)
- 📊 Comparative Specifications (highlight feature matches)
- 💚 Sustainability Highlights
- ⚡ Power Efficiency Notes
- ⚠️ Potential Limitations
- 🔍 Feature Match Summary (how well each matches your requested features)
"""

    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Could not generate analysis: {str(e)}"

def get_product_by_id(pid: str) -> Optional[Dict]:
    """Enhanced product retrieval with multiple fallback methods"""
    try:
        result = collection.get(ids=[pid], include=["metadatas", "documents"])
        if not result or not result["metadatas"]:
            return None
            
        meta = result["metadatas"][0]
        doc = json.loads(result["documents"][0]) if result["documents"] else {}
        
        # Try getting from metadata first, then document, then direct access
        def get_field(field_name):
            return (
                meta.get(field_name) or 
                doc.get(field_name) or
                meta.get(field_name.replace(" ", "")) or  # Try without spaces
                meta.get(field_name.lower()) or  # Try lowercase
                "N/A"
            )
        
        return {
            # Core identifiers
            "product_id": get_field("Product_id"),
            "product_name": get_field("Product Name"),
            "product_type": get_field("Product Type"),
            "company_name": get_field("Company Name"),
            
            # Scores and descriptions
            "green_index": get_field("Green Index"),
            "green_index_description": get_field("Green Index Description"),
            "innovation_score": get_field("Innovation Score"),
            "innovation_score_description": get_field("Innovation Score Description"),
            "datasheet_score": get_field("Datasheet Score"),
            "datasheet_score_description": get_field("Datasheet Score Description"),
            "recyclability_score": get_field("Recyclability Score"),
            "recyclability_description": get_field("Recyclability Description"),
            "supply_chain_risk_score": get_field("Supply Chain Risk Score"),
            "supply_chain_risk_description": get_field("Supply Chain Risk Description"),
            "repairability_index": get_field("Repairability Index"),
            "repairability_description": get_field("Repairability Description"),
            "manufacturer_sustainability_rating": get_field("Manufacturer Sustainability Rating"),
            "manufacturer_sustainability_description": get_field("Manufacturer Sustainability Description"),
            
            # Technical specifications
            "specs": get_field("Specs"),
            "power_consumption": get_field("Power Consumption (mW)"),
            "voltage_range": get_field("Voltage Range (V)"),
            "temperature_range": get_field("Temperature Range (°C)"),
            "pin_count": get_field("Pin Count"),
            
            # Additional metadata
            "full_metadata": meta,
            "full_document": doc
        }
    except Exception as e:
        print(f"Error retrieving product {pid}: {str(e)}")
        return None

def radar_chart(products: List[Dict]) -> go.Figure:
    """Create comparison radar chart"""
    metrics = [
        "green_index", 
        "innovation_score", 
        "recyclability_score",
        "datasheet_score",
        "supply_chain_risk_score",
        "repairability_score"
    ]
    
    fig = go.Figure()
    
    for product in products:
        values = [float(product.get(m, 0)) for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=[m.replace('_', ' ').title() for m in metrics],
            fill='toself',
            name=f"{product['product_name']} ({product['company_name']})"
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        height=600,
        title="Component Comparison Radar Chart"
    )
    return fig

def save_feedback(product_id: str, feedback: str) -> None:
    """Store user feedback with timestamp"""
    try:
        feedback_entry = {
            "product_id": product_id,
            "feedback": feedback,
            "timestamp": datetime.now().isoformat()
        }
        
        if not os.path.exists("feedback"):
            os.makedirs("feedback")
            
        filepath = os.path.join("feedback", "user_feedback.json")
        
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                data = json.load(f)
        else:
            data = []
            
        data.append(feedback_entry)
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
            
    except Exception as e:
        st.error(f"Could not save feedback: {str(e)}")

# === App State ===
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "gemini_response" not in st.session_state:
    st.session_state.gemini_response = ""

# === Main UI ===
tab1, tab2, tab3, tab4 = st.tabs([
    "🛞 Recommendation Engine", 
    "🆚 Compare Products", 
    "📊 Product Profiler",
    "🏭 Manufacturers",
])

with tab1:
    
    # Center-aligned heading
    st.markdown("""
    <style>
    .centered {
        text-align: center;
    }
    </style>
    <h1 class="centered">🌀 Smart Component Finder</h1>
    """, unsafe_allow_html=True)

    user_query = st.text_input(
        "Describe your component needs:",
        value=st.session_state.get("user_query", ""),
        placeholder="E.g.: 'AI-ready MCU with low power consumption for wearables'"
    )

with tab1:
    #st.markdown("### 🌀 Smart Component Finder")
    
    if st.button("🚀 Find Components", use_container_width=True):
        with st.spinner("Analyzing your requirements..."):
            try:
                # Step 1: Keyword extraction
                extracted = extract_keywords_with_gemini(user_query)
                
                st.markdown("### 🧠 Interpreted Requirements")
                st.json(extracted)
                
                # Store the extracted features in session state
                st.session_state.extracted_features = extracted.get("features", "")
                
                # Step 2: Vector search
                results = retrieve_similar_products(extracted)
                
                if not results:
                    st.warning("No exact matches found. Showing similar components...")
                    results = retrieve_similar_products({"product_type": user_query})
                
                if results:
                    st.session_state.recommendations = results
                    # Pass the features to generate_response
                    st.session_state.gemini_response = generate_response(
                        results, 
                        st.session_state.extracted_features
                    )
                else:
                    st.error("Could not find any matching components. Please try different keywords.")
                    
            except Exception as e:
                st.error(f"System error: {str(e)}")

    if st.session_state.gemini_response:
        st.markdown("### 🤖 Recommendation Analysis")
        st.markdown(st.session_state.gemini_response)
        
    if st.session_state.recommendations:
        st.markdown("### 📦 Matching Components")
        for i, item in enumerate(st.session_state.recommendations[:5]):
            with st.expander(f"🏅 #{i+1}: {item['product_name']}"):
                cols = st.columns([3, 2])
                with cols[0]:
                    st.markdown(f"**Type**: {item['product_type']}")
                    st.markdown(f"**Manufacturer**: [{item['company_name']}]({item.get('company_url', '#')})")
                    st.markdown(f"**Price**: ₹{item.get('price', 'N/A')}")
                    
                    st.markdown("### 📋 Specifications")
                    # List of important fields to display
                    important_fields = [
                        'description', 'use_case', 'power_consumption', 
                        'voltage_range', 'temperature_range', 'compliance_standards',
                        'warranty', 'pin_count', 'mtbf_hours', 'lifecycle_stage'
                    ]
                    
                    for field in important_fields:
                        if field in item and item[field] not in [None, 'N/A', '']:
                            st.markdown(f"**{field.replace('_', ' ').title()}**: {item[field]}")
                    
                    st.markdown(f"[Product Page]({item.get('product_url', '#')})")
                    
                with cols[1]:
                    st.metric("♻️ Green Index", item.get('green_index', 'N/A'))
                    st.metric("💡 Innovation Score", item.get('innovation_score', 'N/A'))
                    st.metric("⚡ Power", f"{item.get('power_consumption', 'N/A')}mW")
                    st.metric("⏱️ Lead Time", f"{item.get('lead_time_days', 'N/A')} days")
                
                feedback_cols = st.columns(3)
                with feedback_cols[0]:
                    if st.button(f"👍 Relevant", key=f"up_{item['product_id']}"):
                        save_feedback(item['product_id'], "relevant")
                        st.success("Feedback saved!")
                with feedback_cols[1]:
                    if st.button(f"👎 Not Relevant", key=f"down_{item['product_id']}"):
                        save_feedback(item['product_id'], "not_relevant")
                        st.info("Thanks for helping improve our system!")
                with feedback_cols[2]:
                    if st.button("📊 Compare", key=f"comp_{item['product_id']}"):
                        st.session_state.compare_ids = [item['product_id']]
                        st.switch_page("?tab=Compare Products")

with tab2:
        
    # Center-aligned heading
    st.markdown("""
    <style>
    .centered {
        text-align: center;
    }
    </style>
    <h1 class="centered">🆚 Component Comparison</h1>
    """, unsafe_allow_html=True)
    
    # Get all available products for selection
    product_data = collection.get(include=["metadatas"])
    all_products = [
        {
            "id": pid,
            "label": f"{meta.get('Product Name', 'Unknown')} ({meta.get('Product Type', 'N/A')})",
            "meta": meta
        }
        for pid, meta in zip(product_data["ids"], product_data["metadatas"])
    ]
    
    # Selection interface
    selected = st.multiselect(
        "Select 2-5 components to compare",
        options=[p["label"] for p in all_products],
        max_selections=5
    )
    
    # Get selected product data
    selected_products = []
    for p in all_products:
        if p["label"] in selected:
            product = {
                "product_id": p["id"],
                "product_name": p["meta"].get("Product Name", "Unknown"),
                "company_name": p["meta"].get("Company Name", "N/A"),
                "product_type": p["meta"].get("Product Type", "N/A"),
                "price": p["meta"].get("Price (INR)", "N/A"),
                "green_index": p["meta"].get("Green Index", "N/A"),
                "innovation_score": p["meta"].get("Innovation Score", "N/A"),
                "datasheet_score": p["meta"].get("Datasheet Score", "N/A"),
                "recyclability_score": p["meta"].get("Recyclability Score", "N/A"),
                "repairability_score": p["meta"].get("Repairability Index", "N/A"),
                "packaging_type": p["meta"].get("Packaging Type", "N/A"),
                "power_consumption": p["meta"].get("Power Consumption (mW)", "N/A"),
                "supply_chain_risk_score": p["meta"].get("Supply Chain Risk Score", "N/A"),
                "specs": p["meta"].get("Specs", "N/A") 
            }
            selected_products.append(product)
    
    # Comparison visualization
    if len(selected_products) >= 2:
        st.plotly_chart(radar_chart(selected_products), use_container_width=True)
        
        # Detailed comparison table
        st.markdown("### 📊 Specification Comparison")
        
        # Prepare the comparison data
        # Prepare the comparison data
        comparison_data = []
        for product in selected_products:
            row = {
        "Product Name": product["product_name"],
        "Manufacturer": product["company_name"],
        "Type": product["product_type"],
        "Price (₹)": f"₹{product['price']}" if isinstance(product["price"], (int, float)) else product["price"],
        "Green Index": product["green_index"],
        "Innovation Score": product["innovation_score"],
        "Datasheet Score": product["datasheet_score"],
        "Recyclability Score": product["recyclability_score"],
        "Repaiability Score": product["repairability_score"],
        "Packaging": product["packaging_type"],
        "Power (mW)": product["power_consumption"],
        "Supply Chain Risk": product["supply_chain_risk_score"],
        "Features": product.get("specs", "N/A")  # This is the correct field name
        }
            comparison_data.append(row)
        
        # Create and display the dataframe
        df = pd.DataFrame(comparison_data)
        
        # Configure column-specific formatting
        column_config = {
            "Green Index": st.column_config.ProgressColumn(
                format="%.1f",
                min_value=0,
                max_value=10,
                help="Environmental sustainability score (0-10)"
            ),
            "Innovation Score": st.column_config.ProgressColumn(
                format="%.1f", 
                min_value=0,
                max_value=10,
                help="Technical innovation rating (0-10)"
            ),
            "Datasheet Score": st.column_config.ProgressColumn(
                format="%.1f",
                min_value=0,
                max_value=10,
                help="Quality of documentation (0-10)"
            ),
            "Recyclability Score": st.column_config.ProgressColumn(
                format="%.1f",
                min_value=0,
                max_value=10,
                help="Ease of recycling (0-10)"
            ),
            "Supply Chain Risk": st.column_config.ProgressColumn(
                format="%.1f",
                min_value=0,
                max_value=10,
                help="Lower is better (0-10)",
            ),
            "Specifications": st.column_config.TextColumn(
                width="large",
                help="Complete technical specifications from database"
            )
        }
        
        st.dataframe(
            df,
            column_config=column_config,
            use_container_width=True,
            hide_index=True
        )
        
        # Generate comparative analysis
        if st.button("💡 Generate Comparative Analysis"):
            with st.spinner("Generating expert analysis..."):
                # In the comparison section of tab2, replace the context building part with:

                context = "\n".join([
    f"- {row['Product Name']} by {row['Manufacturer']}: "
    f"Type: {row['Type']} | "
    f"Price: {row['Price (₹)']} | "
    f"Green Index: {row['Green Index']} | "
    f"Innovation: {row['Innovation Score']} | "
    f"Datasheet: {row['Datasheet Score']} | "
    f"Recyclability: {row['Recyclability Score']} | "
    f"Packaging: {row['Packaging']} | "
    f"Power: {row['Power (mW)']} | "
    f"Specs: {row.get('Features', 'N/A')}"  # Changed from 'Specs' to 'Features'
    for row in comparison_data
                ])
                
                prompt = f"""
Compare these electronic components considering both technical and sustainability factors:
{context}

Provide a detailed analysis including:
1. Best overall choice considering all factors
2. Most cost-effective option
3. Most sustainable option (considering Green Index, Recyclability Score, and Packaging)
4. Best documented product (based on Datasheet Score)
5. Technical performance comparison (analyze the full specifications)
6. Environmental impact analysis (recyclability and packaging)
7. Power efficiency comparison

Format your response with clear sections and bullet points.
"""
                try:
                    response = llm.invoke(prompt)
                    st.markdown("### 🤖 Comparative Analysis")
                    st.markdown(response.content)
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")

with tab3:
    # Center-aligned heading with custom styling
    st.markdown("""
    <style>
    .centered {
        text-align: center;
        margin-bottom: 30px;
    }
    .specs-heading {
        font-size: 24px !important;
        margin-bottom: 15px !important;
        color: #2c3e50;
    }
    .specs-text {
        font-size: 26px !important;
        line-height: 1.6 !important;
        color: #34495e;
    }
    .specs-item {
        margin-bottom: 10px !important;
        padding-left: 15px;
        text-indent: -15px;
    }
    .specs-container {
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        margin-bottom: 25px;
        border-left: 4px solid #3498db;
    }
    .metric-label {
        font-weight: 600;
        color: #2c3e50;
    }
    .sustainability-metrics {
        font-size: 15px !important;
        line-height: 1.8 !important;
    }
    </style>
    <h1 class="centered">🔍 Component Deep Dive</h1>
    """, unsafe_allow_html=True)

    # Get all products for selector including documents
    product_data = collection.get(include=["metadatas", "documents"])
    all_products = [
        {
            "id": pid,
            "label": f"{meta.get('Product Name', 'Unknown')} ({meta.get('Product Type', 'N/A')})",
            "meta": meta,
            "doc": doc
        }
        for pid, meta, doc in zip(product_data["ids"], product_data["metadatas"], product_data["documents"])
    ]
    
    # Product selection
    selected_profile = st.selectbox(
        "Select component to analyze",
        options=[p["label"] for p in all_products],
        index=None,
        placeholder="Select a component...",
        key="component_select"
    )
    
    if selected_profile:
        selected_id = None
        for p in all_products:
            if p["label"] == selected_profile:
                selected_id = p["id"]
                break
        
        if selected_id:
            selected_product = next((p for p in all_products if p["id"] == selected_id), None)
            
            if selected_product:
                meta = selected_product["meta"]
                doc = selected_product["doc"]
                
                cols = st.columns([1, 2])
                with cols[0]:
                    st.markdown(f"### {meta.get('Product Name', 'Unknown')}")
                    st.markdown(f"**Manufacturer**: {meta.get('Company Name', 'N/A')}")
                    st.markdown(f"**Type**: {meta.get('Product Type', 'N/A')}")
                    if meta.get('Product URL'):
                        st.markdown(f"**Product Page**: [Link]({meta['Product URL']})")
                    
                    # Metrics section
                    st.markdown("---")
                    st.metric("♻️ Green Index", meta.get('Green Index', 'N/A'))
                    st.metric("💡 Innovation Score", meta.get('Innovation Score', 'N/A'))
                    st.metric("💰 Price", f"₹{meta.get('Price (INR)', 'N/A')}")
                    st.metric("⏱️ Lead Time", f"{meta.get('Lead Time (Days)', 'N/A')} days")
                    st.metric("⚡ Power", f"{meta.get('Power Consumption (mW)', 'N/A')} mW")
                    st.metric("♻️ Recyclability Score", meta.get('Recyclability Score', 'N/A'))
                    st.metric("🔧 Repairability Index", meta.get('Repairability Index', 'N/A'))
                    
                    # Full-width Generate Expert Analysis button
                    st.markdown("---")
                
                with cols[1]:
                    # Specifications section with improved formatting
                    st.markdown('<h2 class="specs-heading">📋 Specifications</h2>', unsafe_allow_html=True)
                    
                    # Container for better visual grouping
                    with st.container():
                        st.markdown(
                            f'<div class="specs-container">'
                            f'<div class="specs-text"><strong>{meta.get("Description", "No description available")}</strong></div>'
                            '<div style="margin-top: 15px;"></div>'
                            f'<div class="specs-text">',
                            unsafe_allow_html=True
                        )
                        
                        # Technical specs with improved formatting
                        tech_specs = [
                            ("Power Consumption", f"{meta.get('Power Consumption (mW)', 'N/A')} mW"),
                            ("Voltage Range", f"{meta.get('Voltage Range (V)', 'N/A')} (V)"),
                            ("Temperature Range", f"{meta.get('Temperature Range (°C)', 'N/A')} (°C)"),
                            ("Pin Count", meta.get('Pin Count', 'N/A')),
                            ("Compliance Standards", meta.get('Compliance Standards', 'N/A')),
                            ("Packaging Type", meta.get('Packaging Type', 'N/A')),
                            ("MTBF", f"{meta.get('MTBF (Hours)', 'N/A')} Hrs"),
                            ("AI Ready", "Yes" if str(meta.get('AI Ready', '')).lower() in ['true', 'yes', '1', 'ves'] else "No"),
                            ("Manufacturing Capacity", f"{meta.get('Manufacturing Capacity (Units/Year)', 'N/A')} Units/Year")
                        ]
                        
                        for k, v in tech_specs:
                            st.markdown(
                                f'<div class="specs-item">• <strong>{k}</strong>: {v}</div>',
                                unsafe_allow_html=True
                            )
                        
                        st.markdown('</div></div>', unsafe_allow_html=True)
                    
                    # Sustainability metrics expander with improved formatting
                    with st.expander("📈 Sustainability Metrics", expanded=False):
                        st.markdown("""
                        <style>
                        .sustainability-metrics {
                            font-size: 15px !important;
                            line-height: 1.8 !important;
                        }
                        .metric-item {
                            margin-bottom: 10px;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        sustain_metrics = [
                            ("Carbon Footprint", f"{meta.get('Carbon Footprint (g)', 'N/A')}g"),
                            ("ESG Certified", meta.get('ESG Certified', 'N/A')),
                            ("Supply Chain Risk Score", meta.get('Supply Chain Risk Score', 'N/A')),
                            ("Supply Chain Risk", meta.get('Supply Chain Risk Description', 'N/A')),
                            ("Repairability", meta.get('Repairability Description', 'N/A')),
                            ("Recyclability", meta.get('Recyclability Description', 'N/A')),
                            ("Hazardous Materials", meta.get('Hazardous Materials', 'N/A')),
                            ("Manufacturer Sustainability", meta.get('Manufacturer Sustainability Rating', 'N/A')),
                            ("Manufacturer Sustainability Info", meta.get('Manufacturer Sustainability Description', 'N/A'))
                        ]
                        
                        for k, v in sustain_metrics:
                            st.markdown(
                                f'<div class="metric-item"><strong>{k}:</strong> {v}</div>',
                                unsafe_allow_html=True
                            )
                
                # Full-width Generate Expert Analysis button
                if st.button("🧠 Generate Expert Analysis", 
                            use_container_width=True,
                            key=f"analyze_{selected_id}"):
                    with st.spinner("🧠 Analyzing component... This may take a moment"):
                        try:
                            # Prepare the specifications text for the prompt
                            specs_for_prompt = "\n".join([f"{k}: {v}" for k, v in tech_specs])
                            
                            prompt = f"""
Analyze this electronic component as an electronic semiconductor design and manufacturer expert:

**Component Details:**
- Name: {meta.get('Product Name', 'Unknown')}
- Manufacturer: {meta.get('Company Name', 'N/A')}
- Type: {meta.get('Product Type', 'N/A')}
- Description: {meta.get('Description', 'N/A')}

**Technical Specifications:**
{specs_for_prompt}

**Key Metrics:**
- Price: ₹{meta.get('Price (INR)', 'N/A')}
- Green Index: {meta.get('Green Index', 'N/A')}/10
- Innovation Score: {meta.get('Innovation Score', 'N/A')}/10
- Repairability: {meta.get('Repairability Index', 'N/A')}/10
- Power Consumption: {meta.get('Power Consumption (mW)', 'N/A')}mW
- Recyclability: {meta.get('Recyclability Score', 'N/A')}/10

**Please provide a detailed analysis covering:**
1. Key strengths and competitive advantages
2. Potential weaknesses or limitations
3. Ideal applications and use cases
4. Sustainability evaluation (environmental impact)
5. Price-to-performance assessment
6. Market positioning and alternatives
7. Recommendations for optimal usage

Format your response with clear section headings and bullet points for readability.
"""
                            response = llm.invoke(prompt)
                            
                            # Display the analysis with nice formatting
                            st.markdown("## 📝 Expert Analysis")
                            st.markdown("---")
                            st.markdown(response.content)
                            
                        except Exception as e:
                            st.error(f"⚠️ Analysis failed: {str(e)}")
                            st.info("Please try again or check your connection to the analysis service.")
                
# Helper functions
def display_manufacturers_table(manufacturer_data):
    """Display the manufacturers table with hyperlinks"""
    display_data = []
    for company, stats in manufacturer_data.items():
        manufacturer_display = company
        if stats["company_url"]:
            manufacturer_display = f'<a href="{stats["company_url"]}" target="_blank">{company}</a>'
        
        sample_products = []
        for name, url in zip(stats["products"][:3], stats["product_urls"][:3]):
            if url:
                sample_products.append(f'<a href="{url}" target="_blank">{name}</a>')
            else:
                sample_products.append(name)
        
        display_data.append({
            "Manufacturer": manufacturer_display,
            "Product Count": stats["product_count"],
            "Avg. Green Index": round(np.mean(stats["green_index_avg"]), 1) if stats["green_index_avg"] else "N/A",
            "Avg. Innovation": round(np.mean(stats["innovation_avg"]), 1) if stats["innovation_avg"] else "N/A",
            "Avg. Repairability": round(np.mean(stats.get("repairability_avg", [])), 1) if stats.get("repairability_avg") else "N/A",
            "Avg. Recyclability": round(np.mean(stats.get("recyclability_avg", [])), 1) if stats.get("recyclability_avg") else "N/A",
            "Avg. Datasheet": round(np.mean(stats.get("datasheet_avg", [])), 1) if stats.get("datasheet_avg") else "N/A",
            "Sample Products": ", ".join(sample_products) + ("..." if len(stats["products"]) > 3 else ""),
            "Product Range": f"{len(stats['products'])} models"
        })
    
    display_data.sort(key=lambda x: x["Product Count"], reverse=True)
    st.markdown(
        pd.DataFrame(display_data).to_html(escape=False, index=False),
        unsafe_allow_html=True
    )

def display_radar_chart(selected_companies, manufacturer_data):
    """Display radar chart comparing selected companies"""
    if len(selected_companies) < 2:
        st.warning("Please select at least 2 companies to compare")
        return
    
    metrics = [
        'Green Index',
        'Innovation Score', 
        'Repairability Index',
        'Recyclability Score',
        'Datasheet Score'
    ]
    
    plot_data = []
    for company in selected_companies:
        if company in manufacturer_data:
            stats = {
                'Metric': metrics,
                'Value': [
                    np.mean(manufacturer_data[company]['green_index_avg']) if manufacturer_data[company]['green_index_avg'] else 0,
                    np.mean(manufacturer_data[company]['innovation_avg']) if manufacturer_data[company]['innovation_avg'] else 0,
                    np.mean(manufacturer_data[company].get('repairability_avg', [])) if manufacturer_data[company].get('repairability_avg') else 0,
                    np.mean(manufacturer_data[company].get('recyclability_avg', [])) if manufacturer_data[company].get('recyclability_avg') else 0,
                    np.mean(manufacturer_data[company].get('datasheet_avg', [])) if manufacturer_data[company].get('datasheet_avg') else 0
                ],
                'Company': company
            }
            plot_data.append(pd.DataFrame(stats))
    
    if not plot_data:
        st.warning("No data available for selected companies")
        return
    
    df = pd.concat(plot_data)
    
    # Normalize values (0-10 scale)
    for metric in metrics:
        max_val = df[df['Metric'] == metric]['Value'].max()
        if max_val > 0:
            df.loc[df['Metric'] == metric, 'Value'] = (df[df['Metric'] == metric]['Value'] / max_val) * 10
    
    fig = px.line_polar(
        df, 
        r='Value', 
        theta='Metric', 
        color='Company',
        line_close=True,
        template='plotly_dark',
        title='Company Comparison (Normalized Scores)',
        labels={'Value': 'Score (0-12)'}
    )
    
    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 11])),
        showlegend=True,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab4:

    # Center-aligned heading
    st.markdown("""
    <style>
    .centered {
        text-align: center;
    }
    </style>
    <h1 class="centered">🏭 Component Manufacturers Directory</h1>
    """, unsafe_allow_html=True)
    
    # Get all unique product types
    all_products = collection.get(include=["metadatas"])
    product_types = sorted(list(set(
        p["Product Type"] for p in all_products["metadatas"] if p and "Product Type" in p
    )))
    
    # Product type selector
    selected_type = st.selectbox(
        "Search by component type:",
        options=product_types,
        index=0,
        help="Find all manufacturers for a specific component type"
    )
    
    if st.button("🔍 Find All Manufacturers", type="primary",):
        with st.spinner(f"Compiling manufacturer data for {selected_type}..."):
            results = collection.query(
                query_texts=[selected_type],
                n_results=100,
                include=["metadatas", "documents"]
            )
            
            manufacturer_data = {}
            product_details = {}
            
            for meta, doc in zip(results["metadatas"][0], results["documents"][0]):
                if not meta:
                    continue
                
                company = meta.get("Company Name", "Unknown").strip()
                company_url = meta.get("company_url", "").strip()
                product_name = meta.get("Product Name", "").strip()
                product_url = meta.get("Product URL", "").strip()
                
                if not company:
                    continue
                
                if company not in manufacturer_data:
                    manufacturer_data[company] = {
                        "product_count": 0,
                        "products": [],
                        "product_urls": [],
                        "green_index_avg": [],
                        "innovation_avg": [],
                        "repairability_avg": [],
                        "recyclability_avg": [],
                        "datasheet_avg": [],
                        "company_url": company_url
                    }
                
                manufacturer_data[company]["product_count"] += 1
                manufacturer_data[company]["products"].append(product_name)
                manufacturer_data[company]["product_urls"].append(product_url)
                
                if company not in product_details:
                    product_details[company] = []
                product_details[company].append({
                    "name": product_name,
                    "url": product_url
                })
                
                # Process all metrics including the new ones
                for field, target in [
                    ("Green Index", "green_index_avg"),
                    ("Innovation Score", "innovation_avg"),
                    ("Repairability Index", "repairability_avg"),
                    ("Recyclability Score", "recyclability_avg"),
                    ("Datasheet Score", "datasheet_avg")
                ]:
                    if meta.get(field):
                        try:
                            manufacturer_data[company][target].append(float(meta[field]))
                        except (ValueError, TypeError):
                            pass
            
            if manufacturer_data:
                st.session_state.manufacturer_data = manufacturer_data
                st.session_state.product_details = product_details

    # Display manufacturers table
    if hasattr(st.session_state, 'manufacturer_data'):
        with st.expander(f"📊 {len(st.session_state.manufacturer_data)} Manufacturers Found", expanded=True):
            display_manufacturers_table(st.session_state.manufacturer_data)

        # Company comparison section
        st.markdown("---")
                
        # Center-aligned heading
        st.markdown("""
        <style>
        .centered {
        text-align: center;
        }
        </style>
        <h1 class="centered">📊 Compare Companies</h1>
        """, unsafe_allow_html=True)
        
        available_companies = list(st.session_state.manufacturer_data.keys())
        selected_companies = st.multiselect(
            "Select 2 or more companies to compare:",
            options=available_companies,
            default=available_companies[:2] if len(available_companies) >= 2 else [],
            key="company_comparison"
        )
        
        if selected_companies:
            display_radar_chart(selected_companies, st.session_state.manufacturer_data)

    # Company search section
    if hasattr(st.session_state, 'product_details'):
        st.markdown("---")
        
        # Center-aligned heading
        st.markdown("""
        <style>
        .centered {
        text-align: center;
        }
        </style>
        <h1 class="centered">🔍 Search Products by Company</h1>
        """, unsafe_allow_html=True)

        company_search = st.text_input(
            "Enter company name:",
            placeholder="Type a company name to see all their products...",
            help="Search for all products from a specific manufacturer",
            key="company_search_input"
        )
        
        if company_search:
            matching_companies = [
                c for c in st.session_state.product_details.keys() 
                if company_search.lower() in c.lower()
            ]
            
            if matching_companies:
                for company in matching_companies:
                    st.markdown(f"### {company}")
                    product_links = [
                        f'- <a href="{p["url"]}" target="_blank">{p["name"]}</a>' if p["url"] else f'- {p["name"]}'
                        for p in st.session_state.product_details[company]
                    ]
                    st.markdown("\n".join(product_links), unsafe_allow_html=True)
            else:
                st.warning(f"No companies found matching '{company_search}'")

    # Suggested additional data columns
    with st.expander("💡 Recommended Additional Data Fields"):
        st.markdown("""
        To make this directory more powerful, consider adding these fields to your dataset:
        - **Company Metadata**:
          - `company_country`: Manufacturer's headquarters location
          - `company_website`: URL to company site
          - `company_size`: Employee count range
          - `year_founded`: Establishment year
        
        - **Product Details**:
          - `rohs_compliant`: Boolean for environmental compliance
          - `production_locations`: Countries where manufactured
          - `minimum_order_quantity`: For bulk purchases
        
        - **Quality Metrics**:
          - `reliability_score`: 1-10 rating
          - `warranty_period`: Years of coverage
          - `certifications`: ISO, UL, etc.
        """)



# === Footer ===
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
    ESDM Recommender | Rajith Ramachandran | Abhishek Tibrewal | Capstone 2024-2025
    </div>
    """, unsafe_allow_html=True)