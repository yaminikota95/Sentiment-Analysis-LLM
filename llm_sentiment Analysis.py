# -*- coding: utf-8 -*-
"""
Created on Thu May 29 01:06:31 2025

@author: Yamini
"""
from transformers import pipeline
import gradio as gr
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Load Hugging Face sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", framework="pt", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Function to analyze sentiment
def analyze_sentiment(text):
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = round(result['score'], 2)
        return f"{label} (Confidence: {score})"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 📈 Sentiment Analysis Tool (Hugging Face, No API Key Required)")
    review_input = gr.Textbox(label="Enter Customer Review", placeholder="Type a review here...", lines=4)
    sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
    submit_btn = gr.Button("Analyze Sentiment")

    submit_btn.click(fn=analyze_sentiment, inputs=[review_input], outputs=[sentiment_output])

# Launch app
demo.launch()
