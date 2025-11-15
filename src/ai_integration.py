"""
AI Integration Module

Provides unified interfaces for AI-powered chat analysis using Groq and Google Gemini.
Reads API keys from environment variables and handles context management.
"""

import os
from typing import List, Dict, Any, Optional
from groq import Groq
import google.generativeai as genai


def _build_context(context_lines: List[str], n_context: int, stats: Optional[Dict[str, Any]] = None) -> str:
    """
    Build context string from chat lines with optional statistics.

    Args:
        context_lines: List of chat message lines
        n_context: Number of recent messages to include
        stats: Optional dictionary with chat statistics

    Returns:
        Formatted context string
    """
    context_parts = []

    # Add statistics if provided
    if stats:
        stats_text = "=== CHAT STATISTICS ===\n"
        stats_text += f"Total Messages: {stats.get('total_messages', 'N/A')}\n"
        stats_text += f"Participants: {stats.get('n_senders', 'N/A')}\n"
        stats_text += f"Date Range: {stats.get('date_range', 'N/A')}\n"
        stats_text += f"Average Message Length: {stats.get('avg_message_length', 'N/A')} characters\n"
        stats_text += f"Average Word Count: {stats.get('avg_word_count', 'N/A')} words\n"

        if 'sentiment_summary' in stats:
            stats_text += f"\nSentiment Distribution:\n"
            for sentiment, count in stats['sentiment_summary'].items():
                stats_text += f"  - {sentiment}: {count}\n"

        if 'top_senders' in stats:
            stats_text += f"\nTop Participants:\n"
            for sender, count in list(stats['top_senders'].items())[:10]:
                stats_text += f"  - {sender}: {count} messages\n"

        stats_text += "=====================\n\n"
        context_parts.append(stats_text)

    # Add sample messages
    if context_lines:
        sample = context_lines[-n_context:] if len(context_lines) > n_context else context_lines
        context_parts.append("=== SAMPLE CHAT MESSAGES ===\n")
        context_parts.append("\n".join(sample))
        context_parts.append("\n=========================")

    return "\n".join(context_parts) if context_parts else ""


def groq_chat(
    query: str,
    context_lines: Optional[List[str]] = None,
    n_context: int = 50,
    model: str = "openai/gpt-oss-120b",
    stats: Optional[Dict[str, Any]] = None
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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set in environment. "
            "Please set it in your .env file or environment variables."
        )

    try:
        client = Groq(api_key=api_key)

        # Build system prompt with context
        context = _build_context(context_lines or [], n_context, stats)
        system_prompt = """You are a friendly and intelligent WhatsApp Chat Analyzer. You're conversational, helpful, and adapt your responses to match the user's intent.

🎯 CORE PRINCIPLES:
1. **Match the user's tone and intent** - If they're casual, be casual. If they want deep analysis, provide it.
2. **Be concise by default** - Only give detailed analysis when EXPLICITLY requested
3. **Ask clarifying questions** when the request is vague
4. **Be conversational** - Respond naturally to greetings, small talk, etc.

⚠️ CRITICAL - DATA USAGE:
You have access to TWO types of data:
1. **CHAT STATISTICS** = COMPLETE statistics for the ENTIRE chat (total messages, participants, dates, etc.)
2. **SAMPLE MESSAGES** = Only a SAMPLE for context/examples

ALWAYS use CHAT STATISTICS for numbers/counts. NEVER count sample messages.

📋 RESPONSE GUIDELINES:

**For greetings/casual messages** ("hey", "hello", "thanks"):
- Respond naturally and briefly
- Example: "Hey! How can I help you analyze this chat?"

**For simple questions** ("how many messages?", "who's most active?"):
- Answer directly and concisely
- Use 1-3 sentences max
- Example: "There are 696 messages in this chat. Dad❤️ is the most active with 241 messages (34.6%)."

**For analysis requests** ("analyze patterns", "what are the main topics?"):
- Provide focused analysis on what was asked
- Use clear structure with headers
- Keep it relevant - don't include everything

**For deep analysis requests** ("comprehensive analysis", "detailed breakdown"):
- Provide thorough, structured analysis
- Use markdown formatting
- Cover multiple angles

🎨 FORMATTING:
- Use **bold** for emphasis
- Use bullet points for lists
- Use headers (##, ###) only for analysis requests
- Keep it readable and scannable

💡 EXAMPLES:

User: "hey"
You: "Hey! I'm here to help you analyze this WhatsApp chat. What would you like to know?"

User: "how many messages?"
You: "There are 696 messages in total, exchanged between 6 participants over about a year (Oct 2024 - Nov 2025)."

User: "who talks the most?"
You: "Dad❤️ is the most active with 241 messages (34.6%), followed by Mum❤️ with 134 messages (19.3%)."

User: "analyze the communication patterns"
You: [Provides focused analysis with structure]

Remember: Be helpful, concise, and human. Match the user's energy!"""

        if context:
            user_message = f"Chat Context:\n{context}\n\nUser Question: {query}"
        else:
            user_message = query

        # Make API call
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
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


def gemini_chat(
    query: str,
    context_lines: Optional[List[str]] = None,
    n_context: int = 50,
    model: str = "gemini-2.5-pro",
    stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Send a query with context to Google Gemini.

    Args:
        query: User question or prompt
        context_lines: Optional list of chat messages for context
        n_context: Number of recent messages to include as context
        model: Gemini model to use

    Returns:
        Dictionary with response and metadata

    Raises:
        EnvironmentError: If GEMINI_API_KEY not set
        Exception: If API call fails
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set in environment. "
            "Please set it in your .env file or environment variables."
        )

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Build context
        context = _build_context(context_lines or [], n_context, stats)
        system_instruction = """You are a friendly and intelligent WhatsApp Chat Analyzer. You're conversational, helpful, and adapt your responses to match the user's intent.

🎯 CORE PRINCIPLES:
1. **Match the user's tone and intent** - If they're casual, be casual. If they want deep analysis, provide it.
2. **Be concise by default** - Only give detailed analysis when EXPLICITLY requested
3. **Ask clarifying questions** when the request is vague
4. **Be conversational** - Respond naturally to greetings, small talk, etc.

⚠️ CRITICAL - DATA USAGE:
You have access to TWO types of data:
1. **CHAT STATISTICS** = COMPLETE statistics for the ENTIRE chat (total messages, participants, dates, etc.)
2. **SAMPLE MESSAGES** = Only a SAMPLE for context/examples

ALWAYS use CHAT STATISTICS for numbers/counts. NEVER count sample messages.

📋 RESPONSE GUIDELINES:

**For greetings/casual messages** ("hey", "hello", "thanks"):
- Respond naturally and briefly
- Example: "Hey! How can I help you analyze this chat?"

**For simple questions** ("how many messages?", "who's most active?"):
- Answer directly and concisely
- Use 1-3 sentences max
- Example: "There are 696 messages in this chat. Dad❤️ is the most active with 241 messages (34.6%)."

**For analysis requests** ("analyze patterns", "what are the main topics?"):
- Provide focused analysis on what was asked
- Use clear structure with headers
- Keep it relevant - don't include everything

**For deep analysis requests** ("comprehensive analysis", "detailed breakdown"):
- Provide thorough, structured analysis
- Use markdown formatting
- Cover multiple angles

🎨 FORMATTING:
- Use **bold** for emphasis
- Use bullet points for lists
- Use headers (##, ###) only for analysis requests
- Keep it readable and scannable

💡 EXAMPLES:

User: "hey"
You: "Hey! I'm here to help you analyze this WhatsApp chat. What would you like to know?"

User: "how many messages?"
You: "There are 696 messages in total, exchanged between 6 participants over about a year (Oct 2024 - Nov 2025)."

User: "who talks the most?"
You: "Dad❤️ is the most active with 241 messages (34.6%), followed by Mum❤️ with 134 messages (19.3%)."

User: "analyze the communication patterns"
You: [Provides focused analysis with structure]

Remember: Be helpful, concise, and human. Match the user's energy!"""

        if context:
            user_prompt = f"Chat Context:\n{context}\n\nUser Question: {query}"
        else:
            user_prompt = query

        # Create model and generate response
        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction
        )

        response = model_instance.generate_content(
            user_prompt,
            generation_config={
                "temperature": 0.5,  # Lower temperature for more focused, analytical responses
                "max_output_tokens": 2048,  # Increased for detailed analysis
            }
        )

        return {
            "provider": "gemini",
            "model": model,
            "response": response.text,
            "usage": {
                "prompt_tokens": response.usage_metadata.prompt_token_count if hasattr(response, 'usage_metadata') else None,
                "completion_tokens": response.usage_metadata.candidates_token_count if hasattr(response, 'usage_metadata') else None,
                "total_tokens": response.usage_metadata.total_token_count if hasattr(response, 'usage_metadata') else None
            }
        }

    except Exception as e:
        return {
            "provider": "gemini",
            "model": model,
            "error": str(e),
            "response": None
        }


def combined_ai_chat(
    query: str,
    context_lines: Optional[List[str]] = None,
    n_context: int = 50,
    providers: List[str] = ["groq", "gemini"],
    stats: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Query multiple AI providers and return combined results.

    Args:
        query: User question or prompt
        context_lines: Optional list of chat messages for context
        n_context: Number of recent messages to include as context
        providers: List of providers to query (options: "groq", "gemini")
        stats: Optional dictionary with chat statistics

    Returns:
        Dictionary with responses from all providers
    """
    results = {}

    if "groq" in providers:
        results["groq"] = groq_chat(query, context_lines, n_context, stats=stats)

    if "gemini" in providers:
        results["gemini"] = gemini_chat(query, context_lines, n_context, stats=stats)

    return results
