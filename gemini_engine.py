# === STEP 3: Gemini Recommendation Engine ===

from langchain_google_genai import ChatGoogleGenerativeAI


llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key="AIzaSyBgNyOek4g4RhmY_-JG_lxG1n5kA-yKUE8"
)

def generate_response(user_query, products):
    context = "\n\n".join([
        f"- {p['product_name']} ({p['product_type']}): {p['specs']} | Green Index: {p['green_index']}, Innovation: {p['innovation_score']}, Use: {p['use_case']}" 
        for p in products
    ])

    prompt = f"""
    You're an expert in sustainable electronics. Based on the user's query:
    
    "{user_query}"
    
    Recommend and rank the most relevant components from the following list. Highlight why each was chosen:

    {context}
    
    Respond in a professional and helpful tone.
    """

    response = llm.invoke(prompt)
    return response.content
    
def generate_response(prompt: str) -> str:
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"❌ Gemini failed to generate a response: {str(e)}"
