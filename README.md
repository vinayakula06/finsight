# FinSight - AI-Powered Financial Intelligence Platform

[![BuildBharat Through AI Hackathon](https://img.shields.io/badge/Hackathon-BuildBharat_Through_AI-blueviolet)](https://buildbharatthroughai.in/)
[![Prize Money](https://img.shields.io/badge/Prize-₹5000-gold)]
[![A-Hub Support](https://img.shields.io/badge/Investment_Support-A--Hub-brightgreen)](https://www.a-hub.co/)

##   Features

* **Portfolio Management:** Effortlessly track and manage your investment portfolio with real-time valuations and performance analysis.
* **Stock Analysis:** Leverage AI-driven analysis to gain deep insights into stock trends, predictions, and investment opportunities.
* **Trading Tools:** Execute trades efficiently with integrated trading tools and real-time market data.
* **Intelligent AI Assistant:** Get personalized financial advice and support from our AI assistant, answering your queries and providing data-driven recommendations.
* **Market News & Analysis:** Stay informed with fetched and analyzed market news, including AI-driven summaries and sentiment analysis.
* **Smart Stock Recommender:** Receive personalized stock recommendations based on your investment goals, risk tolerance, and other preferences.
* **AI-Based Calling Agent:** Proactive AI-powered voice assistant for personalized user engagement and support, offering financial assistance and platform guidance.

##   Tech Stack

* TensorFlow
* Next.js 14 (Frontend)
* React 18 (Frontend)
* TypeScript (Frontend)
* Tailwind CSS (Frontend)
* Tremor (for charts and UI components - Frontend)
* Headless UI (for accessible components - Frontend)
* Python (Backend)
* Flask (Backend)
* LangChain (Backend - AI Agent for stock recommendations)
* Hugging Face Transformers (Backend - NLP for news analysis)
* MarketAux API (Data Source for news)
* PyTorch (Backend - Stock Prediction Model)
* Vapi.ai (Voice AI)
* OpenAI gpt-4o (LLM for calling agent)
* n8n (Workflow Automation)
* Airtable (Database for calling agent)
* HTML/CSS/JavaScript (Likely for web interfaces)


##   Team

* **AlgoXplorers**

    * Vinay
    * Rohit
    * Sameer
    * Nithin

##   Achievements

* Secured a position in the **Top 5** out of approximately 550 members (110 teams) in the **BuildBharat Through AI** hackathon.
* Awarded **₹5,000** in prize money.
* Received investment support from **A-Hub**.

##   Getting Started

To run FinSight locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>  # Replace <repository_url> with your repo URL
    cd FinSight
    ```

2.  **Install dependencies:**

    * For the Next.js frontend:

        ```bash
        cd finance-app
        npm install
        ```

    * For the Python backend:

        ```bash
        cd stock\ recmondation  # Or market\ analysis for the news tool
        pip install -r requirements.txt  # If you have a requirements.txt
        ```

3.  **Set up environment variables:**

    * You'll need to configure API keys and other environment variables. Create `.env` files as needed.

    * Example for `finance-app/.env`:

        ```
        NEXT_PUBLIC_FINNHUB_API_KEY=YOUR_FINNHUB_KEY
        NEXT_PUBLIC_ALPHA_VANTAGE_API_KEY=YOUR_ALPHA_VANTAGE_API_KEY
        ```

    * Example for `stock recmondation/.env` (or the news tool's directory):

        ```
        GEMINI_API_KEY=YOUR_GEMINI_API_KEY
        MARKETAUX_API_KEY=YOUR_MARKETAUX_API_KEY
        ```

    * **IMPORTANT:** Never commit API keys directly to your repository!

4.  **Run the applications:**

    * For the Next.js frontend:

        ```bash
        cd finance-app
        npm run dev
        ```

    * For the Python backend (stock recommender):

        ```bash
        cd stock\ recmondation
        python app.py  # Or flask run, etc.
        ```

    * For the Python backend (news & analysis tool):

        ```bash
        cd market\ analysis
        python sapp.py
        ```

5.  **Access FinSight in your browser.**

##   Key Components and Usage

###   Finance App (Next.js Frontend)

* Provides the main user interface for interacting with FinSight's features.
* Handles user input and displays financial data, potentially including portfolio information, stock charts, and AI assistant interactions.
* Built using Next.js, React, TypeScript, and Tailwind CSS for a responsive and interactive user experience.
* Fetches data from the backend APIs to populate the UI.

###   Stock Recommender (Python/Flask)

* Located in the `stock recmondation` directory.
* Offers personalized stock recommendations based on user-defined investment preferences.
* **Input:** Users provide their investment goal, risk tolerance, preferred sector, time horizon, and ESG (Environmental, Social, and Governance) importance through a web form.
* **Processing:**
    * A pre-trained PyTorch model predicts stock returns based on historical data.
    * A LangChain AI Agent, powered by the Gemini LLM, generates textual descriptions explaining the rationale behind each recommendation.
* **Output:** The application displays a list of recommended stocks with details like ticker symbol, company name, price, P/E ratio, dividend yield (placeholder data), and a "match percentage" indicating how well the stock aligns with the user's criteria.

###   Market News & Analysis Tool (Python/Flask)

* Located in the `market analysis` directory.
* Aggregates and analyzes market news to provide users with insights.
* **Data Source:** Fetches news articles from the MarketAux API.
* **Processing:**
    * Summarizes news articles using the Hugging Face Transformers library (specifically, the `google-t5/t5-small` model).
    * Performs sentiment analysis on news content to gauge market sentiment (using the `distilbert-base-uncased-finetuned-sst-2-english` model).
* **User Interface:**
    * The home page (`/`) displays recent general market news with pagination.
    * A search form enables users to find and analyze news related to specific topics or ticker symbols.
    * Results are shown with the original news information, AI-generated summaries, and sentiment analysis results.

###   AI-Based Calling Agent

* This component adds a proactive voice interaction capability to FinSight.
* **Objective:** To enhance user engagement by providing personalized support and guidance through AI-powered phone calls.
* **Description:** The FinSight AI Assistant calls users who have recently visited the platform to offer assistance with navigation, financial planning, and investment decisions.
* **Key Components:**
    1.  **User Interaction:** Call scheduling via the FinSight web interface.
    2.  **Workflow Automation (n8n):** Automated workflow to manage user data and trigger calls.
    3.  **Data Management (Airtable):** Airtable database for storing user contact information.
    4.  **Voice AI Platform (Vapi):** Vapi.ai handles call initiation and conversation flow.
    5.  **Conversational Intelligence (OpenAI):** OpenAI's gpt-4o model powers the AI Assistant's conversational abilities and financial knowledge.
* **Technologies Used:**
    * Voice AI: Vapi.ai
    * LLM: OpenAI gpt-4o
    * Workflow Automation: n8n
    * Database: Airtable
    * Frontend: HTML/CSS/JavaScript (likely)
    * Backend: Python/Flask (likely)
* **Highlights:**
    * Integration of multiple APIs and platforms for a seamless calling workflow.
    * Development of an AI persona with specific financial expertise.
    * Application of AI for proactive user support.

##   Contribution

We welcome contributions to FinSight! If you'd like to contribute, please follow these steps:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear, descriptive commit messages.
4.  Submit a pull request.


![projectimg2](https://github.com/user-attachments/assets/e79a4e9e-677b-4568-b85b-ef134268f69b)

![projectimg1](https://github.com/user-attachments/assets/1274bc31-0f5a-4c21-8361-0b8eda211b1a)

![projectimg4](https://github.com/user-attachments/assets/538537fb-7b34-4be5-9b67-bbc34b41d3c9)

![projectimg3](https://github.com/user-attachments/assets/0fde4991-3c81-43a6-98b1-9367a8c60651)



