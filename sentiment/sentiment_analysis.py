# sentiment/sentiment_analysis.py

from newspaper import Article
from textblob import TextBlob
import requests
from datetime import datetime, timedelta

# Sites confiáveis
NEWS_SOURCES = {
    "coindesk": "https://www.coindesk.com",
    "cointelegraph": "https://cointelegraph.com",
}

def get_articles_from_source(base_url, keyword="bitcoin"):
    from newspaper import build
    paper = build(base_url, memoize_articles=False)
    articles = []
    for article in paper.articles[:10]:
        try:
            article.download()
            article.parse()
            if keyword.lower() in article.text.lower():
                articles.append(article)
        except:
            continue
    return articles

def analyze_article_sentiment(article):
    blob = TextBlob(article.text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

def get_news_sentiment(symbol="BTC"):
    keyword = "bitcoin" if symbol == "BTC" else "ethereum"
    total_score = 0
    count = 0

    for source, url in NEWS_SOURCES.items():
        articles = get_articles_from_source(url, keyword)
        for article in articles:
            score = analyze_article_sentiment(article)
            total_score += score
            count += 1

    if count == 0:
        return "neutral"

    avg_score = total_score / count

    if avg_score > 0.1:
        return "bullish"
    elif avg_score < -0.1:
        return "bearish"
    else:
        return "neutral"
