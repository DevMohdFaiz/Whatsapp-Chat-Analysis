import streamlit as st
from groq import Groq
from typing import Dict, Optional, List, Any
from dotenv import get_key


def _get_groq_client() -> Optional['Groq']:
    """Get Groq client instance."""
    if Groq is None:
        return None
    
    api_key = get_key('.env', key_to_get='GROQ_API_KEY')
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment variables. "
            "Please set it in your .env file or environment."
        )

    return Groq(api_key=api_key)


def _call_groq_api(prompt: str, model: str = 'llama-3.3-70b-versatile', max_tokens: int = 1024, 
                   temperature: float = 0.7, retries: int = 3) -> str:
    """
    Internal Groq API wrapper with error handling and retries.

    Args:
        prompt: Prompt to send to the model
        model: Model name (default: llama-3.3-70b-versatile for best quality/speed balance)
               Available models:
               - llama-3.3-70b-versatile: Best for analysis, insights, summaries (280 T/s, high quality)
               - llama-3.1-8b-instant: Fastest for simple tasks (560 T/s, good quality)
               - openai/gpt-oss-120b: Highest quality for complex analysis (500 T/s, premium quality)
               - openai/gpt-oss-20b: Fast high-quality alternative (1000 T/s, good quality)
        max_tokens: Maximum tokens in response
        temperature: Sampling temperature
        retries: Number of retry attempts

    Returns:
        Model response text

    Raises:
        ValueError: If Groq client cannot be initialized
        RuntimeError: If API call fails after retries
    """
    client = _get_groq_client()
    if client is None:
        raise ValueError("Groq client not available. Please install groq package and set GROQ_API_KEY.")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Groq API call failed after {retries} attempts: {str(e)}")
            continue

    raise RuntimeError("Unexpected error in Groq API call")


def build_context(user_prompt, chat_summary, chat_history, max_history_msg: int =5):
    system_prompt = """
    You are a friendly, intelligent WhatsApp Chat Analyzer. You respond conversationally, helpfully, and always adapt your tone to match the user's intent.

    🎯 CORE BEHAVIOR PRINCIPLES
    1. **Match the user’s tone and intention.** If they're casual, be casual. If they ask for deep analysis, provide it.
    2. **Be concise by default.** Only provide detailed analysis when the user explicitly asks for it.
    3. **Ask clarifying questions** when the request is vague or ambiguous.
    4. **Be conversational and natural**—respond appropriately to greetings, thanks, small talk, etc.

    ⚠️ CRITICAL – HOW TO USE DATA
    You will always receive two types of data:

    1. **CHAT STATISTICS (Full Dataset)**  
    - Complete statistics for the entire chat  
    - Use this for ALL numeric information, rankings, counts, and activity summaries  

    2. **SAMPLE MESSAGES (Partial Dataset)**  
    - Small excerpt of chat messages  
    - ONLY use these for examples, tone interpretation, or illustrating patterns  
    - **Never** use them to calculate totals or frequencies  

    📋 RESPONSE RULES

    ### **For Greetings or Casual User Messages**  
    - Keep it short, warm, and natural.  
    Example: “Hey! How can I help you analyze this chat?”

    ### **For Simple Questions**  
    ("how many messages?", "who sent the most texts?", "when was the chat most active?")  
    - Give a direct, concise answer (1–3 sentences).  
    - Rely ONLY on chat statistics.  
    Example: “There are 696 messages in this chat. Dad❤️ is the most active with 241 messages (34.6%).”

    ### **For Analytical Requests**  
    ("analyze communication patterns", "what topics appear?", "how does the tone change?")  
    - Provide structured, focused analysis.  
    - Use bullet points and minimal headers.  
    - Avoid over-explaining unless asked.

    ### **For Deep or Comprehensive Analysis Requests**  
    ("give me a full analysis", "detailed breakdown please")  
    - Provide a rich, multi-layered analysis  
    - Use markdown formatting with headers (##, ###)  
    - Cover multiple angles such as:
    - activity trends  
    - sentiment dynamics  
    - participant behavior  
    - interaction patterns  
    - notable shifts over time  

    🎨 FORMATTING RULES
    - Use **bold** for emphasis  
    - Use bullet points where appropriate  
    - Use headers only for full/deep analysis  
    - Keep responses scannable and well-structured  

    💡 EXAMPLE BEHAVIOR

    User: “hey”  
    You: “Hey! I’m here to help you analyze this WhatsApp chat. What would you like to know?”

    User: “how many messages?”  
    You: “There are 696 total messages exchanged by 6 participants between Oct 2024 and Nov 2025.”

    User: “who talks the most?”  
    You: “Dad❤️ is the most active with 241 messages (34.6%), followed by Mum❤️ with 134 messages (19.3%).”

    User: “analyze the communication patterns”  
    You: *(Provide focused, structured analysis)*

    ---

    Always: be helpful, concise, human, and match the user's energy.


    """
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'system', 'content': chat_summary}
    ]

    if chat_history:
        latest_msg = chat_history[-max_history_msg: ]
        messages.extend(latest_msg)
    messages.append({'role': 'user', 'content': user_prompt})
    return messages


def groq_chat(
    messages: List,
    context_lines: Optional[List[str]] = None,
    n_context: int = 50,
    model: str = "openai/gpt-oss-120b",
    # stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Send a query with context to Groq AI.

    Args:
        query: User question or prompt
        context_lines: Optional list of chat messages for context
        n_context: Number of recent messages to include as context
        model: Groq model to use

    Returns:
        Dictionary with response and metadata

    Raises:
        EnvironmentError: If GROQ_API_KEY not set
        Exception: If API call fails
    """
    try:
        api_key = get_key(".env", key_to_get="GROQ_API_KEY")
    except:
        api_key = st.secrets['GROQ_API_KEY']
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set in environment. "
            "Please set it in your .env file or environment variables."
        )

    try:
        client = Groq(api_key=api_key)


        # Make API call
        chat_completion = client.chat.completions.create(
            messages= messages,
            model=model,
            temperature=0.5,  # Lower temperature for more focused, analytical responses
            max_tokens=2048,  # Increased for detailed analysis
        )

        response_text = chat_completion.choices[0].message.content

        return {
            "provider": "groq",
            "model": model,
            "response": response_text,
            "usage": {
                "prompt_tokens": chat_completion.usage.prompt_tokens,
                "completion_tokens": chat_completion.usage.completion_tokens,
                "total_tokens": chat_completion.usage.total_tokens
            }
        }

    except Exception as e:
        return {
            "provider": "groq",
            "model": model,
            "error": str(e),
            "response": None
        }