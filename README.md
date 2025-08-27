Logical vs. Emotional Chatbot

Overview

This project is a chatbot built using LangGraph and Graph Builder. Unlike traditional chatbots, this one is designed to distinguish between logical and emotional queries and respond accordingly.
By analyzing the nature of the input, it can provide a rational, fact-based response for logical questions or an empathetic, human-like response for emotional ones.
used Gemini API key for this purpose

Features

Query Type Detection: Automatically identifies whether a user's input is logical (fact-based, analytical) or emotional (feelings, opinions, sentiment-driven).

Adaptive Responses: Delivers context-aware responses depending on the type of query.

LangGraph & Graph Builder Integration: Leverages graph-based AI tools for knowledge representation and reasoning, allowing smarter and more structured answers.

Scalable Architecture: Can be extended to handle multiple types of queries or integrate additional reasoning layers.

How It Works

Input Processing: User inputs a query.

Query Classification: The system analyzes the input to determine if it’s logical or emotional.

Graph-Based Reasoning:

For logical queries, the chatbot uses LangGraph to traverse nodes and provide structured, factual answers.

For emotional queries, Graph Builder helps the chatbot craft empathetic, human-like responses.

Response Generation: Outputs a tailored response based on the query type.
