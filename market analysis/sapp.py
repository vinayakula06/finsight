# app.py
import os
import requests
from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
# from dotenv import load_dotenv # Can be removed if not loading any keys from .env
import logging
from typing import Dict, List, Union, Any, Optional
import traceback  # For better error logging
import datetime  # For parsing dates

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO)  # Basic logging

# --- API Keys & URLs ---
MARKETAUX_API_KEY = "G4qN7vEnOdTBRN4MuLAZbXOg7q93fFV9AlB9J38x"
MARKETAUX_NEWS_URL = "https://api.marketaux.com/v1/news/all"
MARKETAUX_WEBSITE_URL = "https://www.marketaux.com/"

app = Flask(__name__)

# --- ML Model Loading ---
summarizer = None
sentiment_analyzer = None
models_loaded = False
tokenizer = None

print("Loading ML models...")
try:
    summarizer_model_name = "google-t5/t5-small"
    sentiment_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=sentiment_model_name)
    print("ML models loaded successfully.")
    models_loaded = True
except Exception as e:
    print(f"CRITICAL: Error loading ML models: {e}")
    traceback.print_exc()
    models_loaded = False


# --- Helper Functions ---
# (fetch_marketaux_news and analyze_content functions remain the same as previous version)
def fetch_marketaux_news(query: str, page: int = 1, articles_per_page: int = 10) -> Dict[str, Any]:
    """Fetches news articles from MarketAux API based on a search query and page number."""
    if not MARKETAUX_API_KEY:
        app.logger.error("MARKETAUX_API_KEY not configured.")
        return {"error": "Server configuration error: Missing MarketAux API Key."}

    search_term = query.strip()
    app.logger.info(
        f"Fetching news from MarketAux for query: '{search_term}', page: {page}, limit: {articles_per_page}")

    try:
        params = {
            'api_token': MARKETAUX_API_KEY,
            'search': search_term,
            'language': 'en',
            'limit': articles_per_page,
            'page': page,
            'sort': 'published_on',
        }

        masked_params = params.copy()
        masked_params['api_token'] = '***'
        request_url = requests.Request(
            'GET', MARKETAUX_NEWS_URL, params=masked_params).prepare().url
        app.logger.info(f"Requesting URL (token masked): {request_url}")

        response = requests.get(MARKETAUX_NEWS_URL, params=params, timeout=15)
        app.logger.info(
            f"MarketAux API Response Status Code: {response.status_code}")
        response.raise_for_status()
        data = response.json()

        if 'data' in data and isinstance(data['data'], list):
            raw_articles = data.get("data", [])
            app.logger.info(
                f"Fetched {len(raw_articles)} articles from MarketAux (Page {page}).")
            if not raw_articles and page == 1:
                app.logger.warning(
                    f"MarketAux returned 0 results for query '{search_term}'.")

            processed_articles = []
            for article in raw_articles:
                content = article.get('description') or article.get(
                    'snippet') or article.get('title', '')
                if not isinstance(content, str):
                    content = str(content) if content is not None else ''

                display_snippet = article.get(
                    'description') or article.get('snippet', '')
                if not isinstance(display_snippet, str):
                    display_snippet = str(
                        display_snippet) if display_snippet is not None else ''

                published_dt_str = "N/A"
                published_at = article.get('published_at')
                if published_at:
                    try:
                        dt_obj = datetime.datetime.fromisoformat(
                            published_at.replace('Z', '+00:00'))
                        published_dt_str = dt_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        app.logger.warning(
                            f"Could not parse MarketAux date: {published_at}")
                        published_dt_str = published_at.split(
                            'T')[0] if 'T' in published_at else published_at

                processed_articles.append({
                    'title': article.get('title', 'No Title'),
                    'link': article.get('url', '#'),
                    'source': article.get('source', 'Unknown Source'),
                    'published_utc': published_dt_str,
                    'content_for_analysis': content,
                    'display_snippet': display_snippet
                })
            return {"articles": processed_articles}
        elif 'error' in data and isinstance(data['error'], dict):
            error_details = data['error']
            error_message = error_details.get(
                "message", "Unknown API error from MarketAux")
            error_code = error_details.get("code", "N/A")
            app.logger.error(
                f"MarketAux API error: Code={error_code}, Message={error_message}")
            return {"error": f"MarketAux News API Error ({error_code}): {error_message}"}
        else:
            app.logger.error(
                f"Unexpected response format from MarketAux: {data}")
            return {"error": "Unexpected response format received from MarketAux."}

    except requests.exceptions.HTTPError as http_err:
        app.logger.error(
            f"HTTP error fetching news from MarketAux: {http_err}")
        try:
            error_details = response.json().get('error', {})
            error_msg = error_details.get(
                'message', f"Status {response.status_code}")
        except:
            error_msg = f"Status {response.status_code}, Body: {response.text[:200]}"
        return {"error": f"HTTP error connecting to MarketAux: {error_msg}"}
    except requests.exceptions.RequestException as e:
        app.logger.error(f"Network error fetching news from MarketAux: {e}")
        return {"error": f"Network or API request error: {e}"}
    except Exception as e:
        app.logger.error(f"Unexpected error fetching news: {e}")
        traceback.print_exc()
        return {"error": f"An unexpected error occurred: {e}"}


def analyze_content(text: str) -> Dict[str, Any]:
    """Analyzes text for summary and sentiment using loaded models."""
    # (This function remains the same)
    if not models_loaded or summarizer is None or sentiment_analyzer is None:
        return {"error": "ML models not available."}
    if not text or not isinstance(text, str):
        return {"summary": "No content provided for analysis.", "sentiment": "N/A"}

    analysis = {}
    try:
        inputs = tokenizer(text, return_tensors="pt",
                           max_length=512, truncation=True, padding=True)
        decoded_input = tokenizer.decode(
            inputs["input_ids"][0], skip_special_tokens=True)
        summary_result = summarizer(
            decoded_input,
            max_length=100,
            min_length=15,
            do_sample=False
        )[0]
        analysis['summary'] = summary_result['summary_text']
    except Exception as e:
        app.logger.error(f"Error during summarization: {e}")
        analysis['summary'] = "Error generating summary."

    try:
        sentiment_result = sentiment_analyzer(text[:512])[0]
        analysis['sentiment'] = sentiment_result['label']
    except Exception as e:
        app.logger.error(f"Error during sentiment analysis: {e}")
        analysis['sentiment'] = "Error analyzing sentiment."

    return analysis

# --- HTML Templates (as strings) ---


# *** MODIFIED HTML_INDEX_TEMPLATE (Filter section REMOVED) ***
HTML_INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market News & Analysis</title>
    <style>
        /* Light Theme based on Finsight Image */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #ffffff; /* White background */
            color: #333; /* Darker text */
        }
        .container {
            background-color: #f8f9fa; /* Light grey container background */
            padding: 25px 35px; /* Adjusted padding */
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); /* Softer shadow */
            max-width: 900px;
            margin: 40px auto;
        }
        h1 {
            color: #212529; /* Dark heading */
            text-align: left;
            margin-bottom: 5px;
            font-size: 1.8em;
            font-weight: 600;
        }
         .subtitle {
            color: #6c757d; /* Medium grey subtitle */
            text-align: left;
            margin-top: 0;
            margin-bottom: 30px;
            font-size: 1em;
            font-weight: 400;
        }
        h2 {
            color: #333;
            border-bottom: 1px solid #dee2e6; /* Lighter border */
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px; /* Increased margin */
            text-align: left;
            font-size: 1.3em; /* Slightly smaller */
            font-weight: 600;
        }
        label {
            font-weight: 600; /* Bolder labels */
            margin-bottom: 8px;
            display: block;
            color: #495057; /* Darker grey label */
            font-size: 0.9em;
        }
        input[type=text] {
            width: 100%;
            padding: 10px 12px;
            margin-bottom: 15px;
            border: 1px solid #ced4da; /* Standard border */
            border-radius: 4px;
            box-sizing: border-box;
            background-color: #ffffff; /* White input */
            color: #495057; /* Dark input text */
            font-size: 1em;
        }
        input[type=text]::placeholder { color: #adb5bd; } /* Lighter placeholder */

        button {
            background-color: #20c997; /* Teal accent */
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            width: 100%;
            transition: background-color 0.2s ease;
        }
        button:hover { background-color: #1baa80; } /* Darker teal */
        .error { color: #dc3545; margin-top: 10px; font-weight: bold; background-color: #f8d7da; border-color: #f5c6cb; padding: 10px; border-radius: 4px; border-left: 3px solid #dc3545; }
        .footer { margin-top: 40px; text-align: center; font-size: 0.9em; color: #6c757d; }
        a { color: #20c997; text-decoration: none; } /* Teal links */
        a:hover { text-decoration: underline; }

        /* News list and item styling */
        .news-list { list-style: none; padding: 0; margin-top: 0;} /* Remove top margin */
        .news-item {
            background-color: #ffffff; /* White card background */
            padding: 15px 20px;
            margin-bottom: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6; /* Light border */
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .news-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;}
        .news-title { font-weight: 600; font-size: 1.1em; margin: 0; flex-grow: 1; padding-right: 15px;}
        .news-title a { color: #333; } /* Dark title link */
        .news-title a:hover { color: #1baa80; }
        .news-date { font-size: 0.8em; color: #6c757d; white-space: nowrap; }
        .news-source { font-size: 0.85em; color: #6c757d; display: block; margin-bottom: 8px; font-weight: 500;}
        .news-description { font-size: 0.95em; color: #495057; margin-bottom: 12px; line-height: 1.5; }
        .analysis { background-color: #e6f9f5; padding: 8px 12px; margin-top: 10px; border-radius: 4px; border-left: 3px solid #20c997; } /* Light teal background */
        .analysis p { margin: 4px 0; font-size: 0.9em; color: #333;}
        .analysis strong { color: #212529; font-weight: 600; }
        .effect-positive { color: #28a745; font-weight: bold; } /* Standard Green */
        .effect-negative { color: #dc3545; font-weight: bold; } /* Standard Red */
        .effect-neutral { color: #ffc107; font-weight: bold; } /* Standard Yellow/Orange */

        /* Load More Button Style */
        .load-more-section { text-align: center; margin-top: 20px; }
        .load-more-button {
            display: inline-block;
            padding: 8px 20px;
            background-color: #6c757d; /* Grey button */
            color: white;
            border-radius: 4px;
            font-weight: 500;
            font-size: 0.9em;
            border: none;
            cursor: pointer;
            transition: background-color 0.2s ease;
        }
        .load-more-button:hover { background-color: #5a6268; text-decoration: none; }

        /* Search Form */
        .search-form-container { padding: 20px 0; margin-top: 20px; border-top: 1px solid #dee2e6;}
        .search-form-container button { width: auto; padding: 10px 20px; font-size: 1em;}
        .search-form-container input[type=text] { margin-bottom: 15px; } /* Ensure margin for button */

        /* Removed Filter Area CSS */

    </style>
</head>
<body>
    <div class="container">
        <h1>Market News & Analysis</h1>
        <p class="subtitle">Latest updates from global financial markets</p>

        <h2>Recent Market News (Page {{ current_page }})</h2>
        {% if initial_error %}
            <p class="error">Error fetching initial news: {{ initial_error }}</p>
        {% elif initial_articles %}
            <div class="news-list">
            {% for article in initial_articles %}
                <div class="news-item">
                    <div class="news-header">
                        <div class="news-title"><a href="{{ article.link }}" target="_blank" rel="noopener noreferrer">{{ article.title }}</a></div>
                        <div class="news-date">{{ article.published_utc }}</div>
                    </div>
                    <div class="news-source">{{ article.source }}</div>
                    <p class="news-description">{{ article.display_snippet }}</p> {# Display snippet/description #}

                    {# Analysis Display Block #}
                    {% if article.analysis and not article.analysis.get('error') %}
                        <div class="analysis">
                            <p><strong>AI Overview:</strong> {{ article.analysis.summary }}</p>
                            <p><strong>Market Effect:</strong>
                                {% set sentiment = article.analysis.sentiment %}
                                {% if sentiment == 'POSITIVE' %}
                                    <span class="effect-positive">Positive Effect</span>
                                {% elif sentiment == 'NEGATIVE' %}
                                    <span class="effect-negative">Negative Effect</span>
                                {% else %}
                                    <span class="effect-neutral">Neutral Effect</span>
                                {% endif %}
                            </p>
                        </div>
                    {% elif article.analysis and article.analysis.get('error') %}
                         <p class="error" style="font-size: 0.9em; padding: 5px; margin-top: 5px;">Analysis Error: {{ article.analysis.error }}</p>
                    {% endif %}
                </div>
            {% endfor %}
            </div>
            {% if initial_articles %} {# Only show Load More if articles were found #}
                <div class="load-more-section">
                     <a href="/?page={{ next_page }}" class="load-more-button">More News</a>
                </div>
            {% endif %}
        {% else %}
            <p>No recent market news found for this page.</p>
        {% endif %}

        <div class="search-form-container">
             <h2>Analyze Specific Topic / Ticker</h2>
             <form action="/analyze" method="POST">
                 <div>
                     <label for="topic">Enter News Topic or Ticker Symbol:</label>
                     <input type="text" id="topic" name="topic" required placeholder="e.g., AAPL, Tesla, AI regulation...">
                 </div>
                 <input type="hidden" name="page" value="1">
                 <div>
                     <button type="submit">Analyze Specific News</button>
                 </div>
             </form>
             {% if search_error %}
                  <p class="error">Search Error: {{ search_error }}</p>
             {% endif %}
        </div>

    </div>
    <div class="footer">
        News potentially via <a href="{{ marketaux_url }}" target="_blank" rel="noopener noreferrer">MarketAux.com</a> | Analysis by Hugging Face models.
    </div>
</body>
</html>
"""

# HTML_RESULTS_TEMPLATE remains the same as the previous version
HTML_RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>News Analysis Results for {{ topic }}</title>
     <style>
        /* Copied styles from index template for consistency */
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #ffffff; /* White background */
            color: #333; /* Darker text */
        }
        .container {
            background-color: #f8f9fa; /* Light grey container background */
            padding: 25px 35px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            max-width: 900px;
            margin: 40px auto;
        }
        h1 {
            color: #212529; /* Dark heading */
            text-align: center;
            margin-bottom: 25px;
            font-weight: 600;
        }
        h2 {
            color: #333;
            border-bottom: 1px solid #dee2e6; /* Lighter border */
            padding-bottom: 10px;
            margin-top: 30px;
            margin-bottom: 20px;
            text-align: left;
            font-size: 1.3em; /* Slightly smaller */
            font-weight: 600;
        }
        .article {
            background-color: #ffffff; /* White card background */
            padding: 15px 20px;
            margin-bottom: 15px;
            border-radius: 6px;
            border: 1px solid #dee2e6; /* Light border */
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .article-title {
            font-weight: 600;
            font-size: 1.1em;
            margin-bottom: 5px;
            color: #333; /* Dark title */
        }
        .article-meta {
            font-size: 0.8em;
            color: #6c757d; /* Medium grey */
            margin-bottom: 10px;
        }
        .article-link {
            font-size: 0.9em;
            color: #20c997; /* Teal links */
            text-decoration: none;
            word-break: break-all;
        }
        .article-link:hover { text-decoration: underline; }
        .analysis {
            background-color: #e6f9f5; /* Light teal background */
            padding: 8px 12px;
            margin-top: 12px;
            border-radius: 4px;
            border-left: 3px solid #20c997; /* Teal border */
        }
        .analysis p { margin: 4px 0; font-size: 0.9em; color: #333;}
        .analysis strong { color: #212529; font-weight: 600;}
        .error { color: #dc3545; font-weight: bold; background-color: #f8d7da; border-color: #f5c6cb; padding: 10px; border-radius: 4px; border-left: 3px solid #dc3545; margin: 10px 0; }
        .footer { margin-top: 40px; text-align: center; font-size: 0.9em; color: #6c757d; }
        a { color: #20c997; text-decoration: none; } /* Teal links */
        a:hover { text-decoration: underline; }

        /* Search form and load more buttons */
        .search-again-form { margin-top: 30px; padding-top: 20px; border-top: 1px solid #dee2e6; }
        .search-again-form label { font-weight: 600; margin-bottom: 8px; display: block; color: #495057;}
        .search-again-form input[type=text] {
             width: calc(100% - 120px); padding: 10px; border: 1px solid #ced4da; border-radius: 4px;
             box-sizing: border-box; display: inline-block; vertical-align: middle;
             background-color: #ffffff; color: #495057; font-size: 1em;
        }
         .search-again-form input[type=text]::placeholder { color: #adb5bd; }
        .search-again-form button {
            background-color: #6c757d; color: white; padding: 10px 15px; border: none; border-radius: 4px;
            cursor: pointer; font-size: 14px; width: 100px; display: inline-block; vertical-align: middle; margin-left: 10px;
            transition: background-color 0.2s ease;
        } /* Grey search button */
        .search-again-form button:hover { background-color: #5a6268; }
        .load-more-form { text-align: center; margin-top: 20px; }
        .load-more-form button {
            background-color: #28a745; color: white; padding: 10px 25px; border: none; border-radius: 4px;
            cursor: pointer; font-size: 14px; transition: background-color 0.2s ease;
        } /* Green load more */
        .load-more-form button:hover { background-color: #218838; }

        /* Sentiment colors */
        .effect-positive { color: #28a745; font-weight: bold; } /* Standard Green */
        .effect-negative { color: #dc3545; font-weight: bold; } /* Standard Red */
        .effect-neutral { color: #ffc107; font-weight: bold; } /* Standard Yellow/Orange */
     </style>
</head>
<body>
    <div class="container">
        <h1>News Analysis Results for "{{ topic }}"</h1>
        <h2>Page {{ current_page }}</h2>

        {% if error %}
            <p class="error">Error fetching or processing news: {{ error }}</p>
        {% elif articles %}
            {% for article in articles %}
            <div class="article">
                <div class="news-header">
                    <div class="news-title">{{ article.title }}</div>
                    <div class="news-date">{{ article.published_utc }}</div>
                </div>
                 <div class="news-source" style="margin-bottom: 10px;">{{ article.source }} | <a href="{{ article.link }}" target="_blank" rel="noopener noreferrer" class="article-link">Read Original</a></div>
                 <p class="news-description">{{ article.display_snippet }}</p>

                {% if article.analysis and not article.analysis.get('error') %}
                    <div class="analysis">
                        <p><strong>AI Overview:</strong> {{ article.analysis.summary }}</p>
                        <p><strong>Market Effect:</strong>
                            {% set sentiment = article.analysis.sentiment %}
                            {% if sentiment == 'POSITIVE' %}
                                <span class="effect-positive">Positive Effect</span>
                            {% elif sentiment == 'NEGATIVE' %}
                                <span class="effect-negative">Negative Effect</span>
                            {% else %}
                                <span class="effect-neutral">Neutral Effect</span>
                            {% endif %}
                        </p>
                    </div>
                {% elif article.analysis and article.analysis.get('error') %}
                     <p class="error" style="font-size: 0.9em; padding: 5px; margin-top: 5px;">Analysis Error: {{ article.analysis.error }}</p>
                {% else %}
                    <p class="error">Could not analyze this article.</p>
                {% endif %}
            </div>
            {% endfor %}

            {% if articles %}
            <div class="load-more-form">
                <form action="/analyze" method="POST">
                    <input type="hidden" name="topic" value="{{ topic }}">
                    <input type="hidden" name="page" value="{{ next_page }}">
                    <button type="submit">Load More Articles (Page {{ next_page }})</button>
                </form>
            </div>
            {% endif %}

        {% else %}
             <p>No news articles found for the specified topic/ticker via MarketAux.com (Page {{ current_page }}).</p>
        {% endif %}

        <div class="search-again-form">
             <h2>Search Again</h2>
             <form action="/analyze" method="POST">
                 <div>
                     <label for="new_topic">Enter New Topic or Ticker Symbol:</label>
                     <input type="text" id="new_topic" name="topic" required placeholder="e.g., GOOG, Microsoft, interest rates...">
                     <input type="hidden" name="page" value="1">
                     <button type="submit">Search</button>
                 </div>
             </form>
        </div>

        <hr style="margin-top: 30px; border-color: #dee2e6;">
        <p style="text-align: center;"><a href="/">Back to Home Page</a></p>
    </div>

    <div class="footer">
        News potentially via <a href="{{ marketaux_url }}" target="_blank" rel="noopener noreferrer">MarketAux.com</a> | Analysis by Hugging Face models.
    </div>
</body>
</html>
"""


# --- Flask Routes ---
# (Routes remain the same as previous version)
@app.route('/', methods=['GET'])
def home():
    """Fetches general market news (paginated), analyzes it, and displays the main page."""
    initial_articles = []
    processed_initial_articles = []
    error_msg = None
    try:
        current_page = request.args.get('page', 1, type=int)
        if current_page < 1:
            current_page = 1
    except ValueError:
        current_page = 1

    if not models_loaded:
        error_msg = "ML Models failed to load. Analysis functionality may be limited."
        news_result = fetch_marketaux_news(
            query="market news OR finance OR economy", page=current_page, articles_per_page=5)
        if "error" in news_result:
            error_msg = news_result["error"]
        else:
            processed_initial_articles = news_result.get("articles", [])
            for article in processed_initial_articles:
                article['analysis'] = {"error": "ML models not loaded."}
    else:
        news_result = fetch_marketaux_news(
            query="market news OR finance OR economy", page=current_page, articles_per_page=5)
        if "error" in news_result:
            error_msg = news_result["error"]
        else:
            articles_to_process = news_result.get("articles", [])
            for article in articles_to_process:
                content_to_analyze = article.get('content_for_analysis', '')
                if content_to_analyze:
                    analysis = analyze_content(content_to_analyze)
                    article['analysis'] = analysis
                else:
                    article['analysis'] = {"error": "No content for analysis."}
                processed_initial_articles.append(article)

    next_page = current_page + 1

    return render_template_string(
        HTML_INDEX_TEMPLATE,
        initial_articles=processed_initial_articles,
        initial_error=error_msg,
        marketaux_url=MARKETAUX_WEBSITE_URL,
        current_page=current_page,
        next_page=next_page
    )


@app.route('/analyze', methods=['POST'])
def analyze_news_route():
    """Handles form submission for specific topic/ticker searches and pagination,
       fetches news from MarketAux, analyzes, and displays results."""
    if not models_loaded:
        return render_template_string(
            HTML_INDEX_TEMPLATE, search_error="ML Models are not available.",
            marketaux_url=MARKETAUX_WEBSITE_URL, current_page=1, next_page=2
        )

    query_input = request.form.get('topic')
    try:
        current_page = int(request.form.get('page', 1))
        if current_page < 1:
            current_page = 1
    except ValueError:
        current_page = 1

    if not query_input:
        return render_template_string(
            HTML_INDEX_TEMPLATE, search_error="Please enter a news topic or ticker.",
            marketaux_url=MARKETAUX_WEBSITE_URL, current_page=1, next_page=2
        )

    articles_per_page = 10
    news_result = fetch_marketaux_news(
        query_input, page=current_page, articles_per_page=articles_per_page
    )

    if "error" in news_result:
        return render_template_string(
            HTML_RESULTS_TEMPLATE, topic=query_input, error=news_result["error"],
            marketaux_url=MARKETAUX_WEBSITE_URL, current_page=current_page, next_page=current_page + 1
        )

    processed_articles = []
    articles = news_result.get("articles", [])

    for article in articles:
        content_to_analyze = article.get('content_for_analysis', '')
        if content_to_analyze:
            analysis = analyze_content(content_to_analyze)
            article['analysis'] = analysis
        else:
            article['analysis'] = {
                "error": "No description/snippet available for analysis."}
        processed_articles.append(article)

    next_page = current_page + 1

    return render_template_string(
        HTML_RESULTS_TEMPLATE, topic=query_input, articles=processed_articles,
        marketaux_url=MARKETAUX_WEBSITE_URL, current_page=current_page, next_page=next_page
    )


# --- Main Execution ---
if __name__ == '__main__':
    if not models_loaded:
        logging.warning(
            "ML Models failed to load. The /analyze endpoint will return errors.")
    app.run(host='0.0.0.0', port=5001, debug=True)
