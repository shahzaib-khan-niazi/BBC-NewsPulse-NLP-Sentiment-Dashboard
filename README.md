# BBC News NLP Dashboard

## Overview
This project is an interactive Python and Streamlit dashboard that analyzes BBC news articles.
It performs sentiment analysis, visualizes word frequency and trends, and uses an SVM model to predict the mood of news articles.
The dashboard allows users to explore news by keyword, sentiment, and time trends, providing insights into the tone and content of news articles.

## Features
 - Interactive Filtering: Search news by keyword and filter by sentiment (Positive, Neutral, Negative)
 - Sentiment Analysis: Classifies articles using VADER sentiment scoring
 - Word Frequency & N-grams: Visualizes most common words and word pairs
 - Trend Visualization: Daily article counts and sentiment trends
 - Predictive Model: LinearSVC model predicts the mood of new news text
 - Word Cloud: Visual representation of top words
 - Interactive Charts: Uses Altair for clean, interactive visualizations

## Technologies Used
 - Python
 - pandas, NumPy
 - Streamlit - NLTK (tokenization, lemmatization, stopwords)
 - scikit-learn (LinearSVC, TF-IDF)
 - Matplotlib, Altair
 - WordCloud




## Usage
1. Upload a BBC news CSV file via the dashboard.
2. Use the sidebar to filter articles by keyword and sentiment.
3. Explore tabs for:
   - Summary of article word counts
   - Most frequent words and word clouds
   - Common word pairs
   - Article sentiment and mood trends
   - Daily article counts and keyword trends
   - Predict mood of your own text using SVM

## Project Highlights
 - Handles large datasets efficiently with pandas and TF-IDF vectorization
 - Combines NLP preprocessing, sentiment scoring, and predictive modeling
 - Fully interactive dashboard suitable for exploratory analysis and presentation


