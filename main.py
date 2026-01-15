import pandas as pd
import numpy as np

bbc_news = pd.read_csv('bbc_news.csv')
print("\ncleaning the data")
print("\n")
print("checking for null values\n",bbc_news.isnull())
print("\n",bbc_news.isnull().sum())
print("\nchecking for duplicate\n",bbc_news.duplicated())
print("\n",bbc_news.duplicated().sum())


import streamlit as st

#streamlit run main.py

st.set_page_config(
    page_title="BBC News Dashboard",
    page_icon="📰",
    layout="wide" #full screen
)
st.sidebar.header("📊 Dashboard Overview")
st.sidebar.markdown("""
Explore BBC news articles interactively:
- Search news by keyword
- Sentiment analysis
- Word frequency & n-grams
- Time trends
- SVM prediction
""")

import re #text manipulation (removing unwanted characters like punctuation, numbers, or extra spaces.)
import nltk #Natural Language Toolkit
from nltk.corpus import stopwords #remove common words like the, is, and, in(not add much meaning).
from nltk.stem import WordNetLemmatizer, PorterStemmer#This helps treat similar words as the same.
#PorterStemmer: Reduces words to their root form,running  run
#WordNetLemmatizer: Converts words to their meaningful base form, better good

from nltk.sentiment import SentimentIntensityAnalyzer #Determines whether text is positive, negative, or neutral, along with a sentiment score.
from nltk.util import ngrams #Determines whether text is positive, negative, or neutral.
from sklearn.feature_extraction.text import TfidfVectorizer # used to convert a text (sentence) documents into a matrix (numerical)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC #Support Vector Classification for predicting classes, adv version of svm
from collections import Counter #Used to count word frequency.
from wordcloud import WordCloud #Creates a visual representation of word frequency.
import altair as alt # Used to create clean, interactive charts that work well with Streamlit dashboards.

#nltk.download('punkt') #split text into sentences and words (tokenization).
#nltk.download('stopwords') #remove unimportant words from text.
#nltk.download('wordnet') #Needed for lemmatization
#nltk.download('vader_lexicon') #Required for sentiment analysi

uploaded_file = st.file_uploader(
    "Upload BBC NEWS csv file ",
    type=['csv']
)

if uploaded_file:

    news_data = pd.read_csv(uploaded_file)
    news_data['pubDate'] = pd.to_datetime(news_data['pubDate'], errors='coerce') #If a value cannot be converted into a valid date,Pandas will replace it with NaT

    # Text Preprocessing

    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english')) #Loads a list of common English words and stores them in a set

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', '', text) #Removes everything except letters and spaces,\s (space),.sub(subsitute)
        tokens = nltk.word_tokenize(text) #Splits text into individual words
        tokens = [
            stemmer.stem(lemmatizer.lemmatize(t))
            for t in tokens if t not in stop_words #Removes common words like the, is, and
        ]
        return tokens

    news_data['processed_words'] = news_data['description'].apply(preprocess)
    news_data['clean_text'] = news_data['processed_words'].apply(lambda x: ' '.join(x)) #Joins all items in a list into one string,Adds a space between each word
    news_data['article_length'] = news_data['processed_words'].apply(len) #Counts how many cleaned words each article contains


    # Sentiment Analysis

    sia = SentimentIntensityAnalyzer()
    def get_sentiment(text):
        score = sia.polarity_scores(text)['compound'] #['compound'] just picks the overall sentiment number from the scores.
        if score >= 0.05:
            return "Positive"
        elif score <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    news_data['sentiment'] = news_data['description'].apply(get_sentiment)

    # Sidebar Filters

    keyword_filter = st.sidebar.text_input("Search News by Keyword")
    sentiment_filter = st.sidebar.multiselect(
        "Filter by Sentiment",
        ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"] #default=[...] means all three options are selected initially
    )

    filtered_data = news_data[
        news_data['sentiment'].isin(sentiment_filter) #[]filters
    ]
    if keyword_filter:
        filtered_data = filtered_data[
            filtered_data['description'].str.contains(keyword_filter, case=False) #case=False, ignores uppercase/lowercase
        ]


    # Metrics

    st.title("📰 BBC News NLP Dashboard")
    col1, col2, col3 = st.columns(3) #Creates three separate containers (col1, col2, col3) side by side
    col1.metric("Number of News Articles", len(filtered_data))
    col2.metric("Total Words in News", filtered_data['article_length'].sum())
    col3.metric("Average Words per News", round(filtered_data['article_length'].mean(),2)) #2 decimal places


    # Prepare Data

    all_words = [w for tokens in filtered_data['processed_words'] for w in tokens] #all_words is a single list of all words from all articles
    freq_d = pd.DataFrame(Counter(all_words).most_common(20), columns=['Word', 'Count']) #Counts how many times each word appears, find top 20 and covert them into dataframe

    bigrams_list = [ng for tokens in filtered_data['processed_words'] for ng in ngrams(tokens,2)] #creates a flat list of all consecutive word pairs (bigrams) from all articles.
    bigram_d = pd.DataFrame(Counter(bigrams_list).most_common(20), columns=['Words','Count']) #counts how many times each bigram appears
    bigram_d['Words'] = bigram_d['Words'].apply(lambda x: ' '.join(x)) #converts each bigram tuple into a readable string by joining the two words with a space.

    daily_counts = filtered_data.groupby(filtered_data['pubDate'].dt.date).size()#extracts just the date part (ignores time),.size() → counts how many articles were published each day
    sentiment_daily = filtered_data.groupby([filtered_data['pubDate'].dt.date, 'sentiment']).size().unstack(fill_value=0)#Groups by both date and sentiment,turns sentiment values into columns and fills missing combinations with 0

    top_words_time = [w for w,c in Counter(all_words).most_common(5)]
    keyword_trends = {} #creates an empty dictionary to store word trends.
    for word in top_words_time:#for each top word, counts how many times it appears each day and stores it in the dictionary with the word as the key.
        keyword_trends[word] = filtered_data[filtered_data['description'].str.contains(word, case=False)].groupby(filtered_data['pubDate'].dt.date).size()
    keyword_trends_df = pd.DataFrame(keyword_trends).fillna(0)#.fillna(0) → replaces missing values (days where the word didn’t appear) with 0


    # Train SVM

    X = news_data['clean_text']
    y = news_data['sentiment']
    #transforms text into a matrix of TF-IDF scores, which shows how important each word is in each document
    vectorizer = TfidfVectorizer(max_features=5000) #keeps only the top 5000 most important word
    X_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42, stratify=y)
    #fixes the randomness so every time you split the data, you get the same train and test sets.
    #makes sure the proportion of each class (like Positive, Neutral, Negative) stays the same in both train and test sets.

    svc_model = LinearSVC()
    svc_model.fit(X_train, y_train)
    svm_acc = svc_model.score(X_test, y_test)

    # Tabs

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Summary","Words","Word Pairs","Mood","Daily Trends","How This News Feels"
    ])

    # Summary
    with tab1:
        st.subheader("Number of Words in Each News Article")
        d_len = filtered_data['article_length'].value_counts().reset_index() #number of words in each article,.value_counts() → counts how many articles have each length,.reset_index() → converts the result into a DataFrame
        d_len.columns = ['Words','Number of Articles'] # Renames columns for clarity: 'Words' = article length, 'Number of Articles' = count
        chart = alt.Chart(d_len).mark_bar().encode(
            x=alt.X('Words', title='Number of Words'),
            y=alt.Y('Number of Articles', title='Number of Articles'),
            tooltip=['Words','Number of Articles']
        ).interactive()
        st.altair_chart(chart, width='stretch')#dth='stretch' → makes the chart fill the available space

    #  Words
    with tab2:
        st.subheader("Most Frequent Words")
        chart = alt.Chart(freq_d).mark_bar().encode(
            x=alt.X('Word', title='Words'),
            y=alt.Y('Count', title='Frequency'),
            tooltip=['Word','Count']
        ).interactive()
        st.altair_chart(chart, width='stretch')

        st.subheader("Word Cloud")
        wc = WordCloud(width=500,height=300,background_color='white').generate(' '.join(all_words))
        st.image(wc.to_array(), width=700)

    # Word Pairs
    with tab3:
        st.subheader("Most Common Word Pairs")
        chart = alt.Chart(bigram_d).mark_bar().encode(
            x=alt.X('Words', title='Word Pairs'),
            y=alt.Y('Count', title='Frequency'),
            tooltip=['Words','Count']
        ).interactive()
        st.altair_chart(chart, width='stretch')

    # Mood
    with tab4:
        st.subheader("Mood of News Articles")
        d_sent = filtered_data['sentiment'].value_counts().reset_index()
        d_sent.columns = ['Mood','Number of Articles']
        d_sent['Color'] = d_sent['Mood'].apply(lambda x:'green' if x=='Positive' else 'blue' if x=='Neutral' else 'red')
        chart = alt.Chart(d_sent).mark_bar().encode(
            x=alt.X('Mood', title='Mood'),
            y=alt.Y('Number of Articles', title='Number of Articles'),
            color=alt.Color('Color', scale=None),
            tooltip=['Mood','Number of Articles']
        ).interactive()
        st.altair_chart(chart, width='stretch')

        st.subheader("Mood Over Time (7-day average)")
        st.line_chart(sentiment_daily.rolling(7).mean())

    # Daily Trends
    with tab5:
        st.subheader("Number of News Articles Each Day")
        st.line_chart(daily_counts)

        st.subheader("Mood of News Each Day")
        st.line_chart(sentiment_daily)

        st.subheader("Top Words Over Time")
        st.line_chart(keyword_trends_df)

    # SVM Prediction
    with tab6:
        st.subheader("Predict Mood of Your News (SVM)")
        st.write(f"Model Accuracy: {round(svm_acc*100,2)}%")
        user_text = st.text_area("Enter news text", height=150)
        if st.button("Predict Mood"):
            if user_text.strip() == "":
                st.warning("Please enter text first!")
            else:
                tokens = preprocess(user_text)
                clean_input = ' '.join(tokens)
                vec_input = vectorizer.transform([clean_input])
                pred = svc_model.predict(vec_input)[0]
                st.success(f"Predicted Mood: **{pred}**")

