# app_pytorch.py

from flask import Flask, request, render_template # No jsonify needed if rendering HTML
from flask_cors import CORS
import torch
import torch.nn as nn # Need nn for model definition
import numpy as np
import pandas as pd
import pickle
import os
# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent # Using ReAct agent framework
from langchain import hub # To pull standard agent prompts
# Standard library imports
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
import json
import re # Import regular expressions for parsing

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Directly assigning your Gemini API key ---
# WARNING: Hardcoding API keys is a security risk. Consider using environment variables
# or a secret management system in production.
GEMINI_API_KEY = "AIzaSyCdXSnAiUrfA1Eas4DPMppGvFbFbL1PSvY" # Replace with your actual key if testing
# ---------------------------------------------

# --- LangChain LLM and Agent Initialization ---
llm = None
agent_executor = None
model_name_to_use = 'gemini-1.5-flash-latest'

try:
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        raise ValueError("Gemini API Key not set or is a placeholder.")

    # 1. Initialize LangChain LLM
    llm = ChatGoogleGenerativeAI(
        model=model_name_to_use,
        google_api_key=GEMINI_API_KEY,
        # Optional: Add temperature, top_p etc. if needed
        # temperature=0.7
        )
    logging.info(f"LangChain ChatGoogleGenerativeAI initialized with model '{model_name_to_use}'.")

    # 2. Define Tools (empty for this basic agent)
    # An agent usually needs tools to interact with the outside world.
    # For this specific task, the LLM can generate the response directly.
    # We provide an empty list to satisfy the AgentExecutor structure.
    tools = []

    # 3. Get the Agent Prompt Template
    # We'll use a standard ReAct (Reasoning and Acting) prompt template
    # This template guides the LLM on how to think step-by-step and potentially use tools.
    prompt_template = hub.pull("hwchase17/react") # Pulls a standard ReAct prompt

    # 4. Create the Agent
    # This combines the LLM, the tools, and the prompt template.
    agent = create_react_agent(llm, tools, prompt_template)
    logging.info(f"LangChain ReAct agent created.")

    # 5. Create the Agent Executor
    # This is the runtime for the agent. It takes user input and orchestrates
    # the agent's steps (reasoning, tool use, final answer generation).
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True, # Set to True to see the agent's thought process in logs
        handle_parsing_errors=True # Helps gracefully handle LLM output format issues
        )
    logging.info(f"LangChain Agent Executor initialized.")

except ValueError as ve:
    logging.critical(f"Configuration Error: {ve}")
    llm = None
    agent_executor = None
except Exception as e:
    logging.critical(f"Failed to initialize LangChain components: {e}", exc_info=True)
    llm = None
    agent_executor = None


# --- App Initialization ---
app = Flask(__name__)
CORS(app)

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML form page."""
    template_path = os.path.join(app.template_folder, 'index.html')
    if not os.path.exists(template_path):
        logging.error(f"Template file not found at: {template_path}")
        return "Error: index.html template not found in 'templates' folder.", 404
    return render_template('index.html')

# Modified /recommend endpoint using LangChain Agent
@app.route('/recommend', methods=['POST'])
def recommend():
    """Receives preferences, gets recommendations from LangChain Agent, returns structured data."""
    # Check if LangChain agent was initialized successfully
    if not agent_executor or not llm:
        return jsonify({"error": "LangChain Agent or LLM not initialized. Check API Key and configuration."}), 500

    try:
        # Get data from the submitted form
        investment_goal = request.form.get('investment_goal', 'Any')
        risk_tolerance = request.form.get('risk_tolerance', '5') # Default to moderate
        preferred_sector = request.form.get('preferred_sector', 'Any')
        time_horizon = request.form.get('time_horizon', 'Any')
        esg_importance = request.form.get('esg_importance', '5') # Default to moderate

        logging.info(f"--- Recommendation Request Received ---")
        logging.info(f"   Goal: {investment_goal}, Risk: {risk_tolerance}, Sector: {preferred_sector}, Horizon: {time_horizon}, ESG: {esg_importance}")
        logging.info(f"------------------------------------")

        # --- Construct the Detailed Input for the LangChain Agent ---
        # This is the core instruction/question we pass to the agent's "input"
        user_request_prompt = f"""
        Act as a financial analyst. Based on the following user preferences:
        - Investment Goal: {investment_goal}
        - Risk Tolerance: {risk_tolerance}/10 (1=low, 10=high)
        - Preferred Sector: {preferred_sector}
        - Investment Time Horizon: {time_horizon}
        - ESG Importance: {esg_importance}/10 (1=low, 10=high)

        Please recommend exactly 3 stocks listed on major US exchanges (like NYSE or NASDAQ).
        For each stock, provide:
        1. Ticker Symbol (e.g., AAPL)
        2. Company Name (e.g., Apple Inc.)
        3. A brief description (2-3 sentences) explaining why it fits the preferences.

        IMPORTANT: Format your *final answer* clearly for parsing, starting directly with the first ticker. Use EXACTLY this format, separating entries with '---':
        Ticker: [SYMBOL]
        Name: [Company Name]
        Description: [Brief description here]
        ---
        Ticker: [SYMBOL]
        Name: [Company Name]
        Description: [Brief description here]
        ---
        Ticker: [SYMBOL]
        Name: [Company Name]
        Description: [Brief description here]

        Do not include any introductory sentences like "Here are the recommendations..." in the final answer. Just provide the Ticker/Name/Description blocks directly.
        """

        # --- Call the LangChain Agent Executor ---
        logging.info(f"Sending request to LangChain Agent Executor (Model: {model_name_to_use})...")
        # The agent executor takes a dictionary, typically with an "input" key
        agent_response = agent_executor.invoke({"input": user_request_prompt})

        # The agent's final response is usually in the 'output' key
        llm_text_response = agent_response.get('output', '')

        # Log the raw response from the agent (which might include reasoning steps if verbose=True)
        logging.info(f"Received raw response structure from Agent Executor: {agent_response}")
        logging.info(f"Extracted final output for parsing:\n{llm_text_response}")


        # --- Parse the LLM Response ---
        recommendations = []
        # Regex adjusted slightly to be less sensitive to leading/trailing whitespace around '---'
        stock_blocks = re.findall(
            r"Ticker:\s*(.*?)\s*Name:\s*(.*?)\s*Description:\s*(.*?)(?:\s*---\s*|\Z)",
            llm_text_response,
            re.DOTALL | re.IGNORECASE
            )

        if not stock_blocks:
            logging.warning(f"Could not parse stock recommendations using regex from agent output.")
            # Provide the raw output in the error for debugging
            return jsonify({"error": "Could not parse recommendations from AI Agent.", "raw_response": llm_text_response}), 500

        for block in stock_blocks:
            ticker, name, description = block
            # Basic cleaning of extracted parts
            ticker = ticker.strip()
            name = name.strip()
            description = description.strip()

            # Basic validation to avoid adding empty entries if regex matches unexpectedly
            if not ticker or not name or not description:
                logging.warning(f"Skipping partially matched block: Ticker='{ticker}', Name='{name}'")
                continue

            recommendations.append({
                "ticker": ticker,
                "name": name,
                "description": description,
                # --- Placeholder Financial Data (Same as before) ---
                "match_percent": f"{100 + int(risk_tolerance) * 3}%", # Simple placeholder calculation
                "price": f"${float(abs(hash(ticker)) % 500 + 50):.2f}", # Hash-based placeholder
                "pe_ratio": f"{float(abs(hash(ticker)) % 40 + 15):.1f}", # Hash-based placeholder
                "dividend_yield": f"{float(abs(hash(ticker)) % 30 / 10):.2f}%" # Hash-based placeholder
                # --- End Placeholder Data ---
            })
            if len(recommendations) >= 3: # Ensure we don't exceed 3 recommendations
                break

        if not recommendations:
            logging.error("Failed to extract any valid recommendations after parsing.")
            return jsonify({"error": "AI Agent did not provide recommendations in the expected format.", "raw_response": llm_text_response}), 500
        elif len(recommendations) < 3:
            logging.warning(f"Agent provided only {len(recommendations)} recommendations in the correct format.")
            # Proceed with fewer recommendations if needed, or return error depending on requirements

        # Return the structured data
        return jsonify({"recommendations": recommendations})

    except Exception as e:
        logging.error(f"Error processing /recommend request: {e}", exc_info=True)
        # Check if the error has agent's thought process if available (depends on exception type)
        error_detail = str(e)
        if hasattr(e, 'llm_output'):
            error_detail += f"\nLLM Output: {e.llm_output}"
        if hasattr(e, 'observation'):
            error_detail += f"\nObservation: {e.observation}"

        return jsonify({"error": "An internal server error occurred processing your request.", "details": error_detail}), 500

# ==============================================================
# == END OF /recommend ENDPOINT ==
# ==============================================================

if __name__ == '__main__':
    print("\nStarting Flask app with LangChain Agent for Recommendations...")
    print(f"Gemini Key Configured: {'Yes (Hardcoded)' if GEMINI_API_KEY and GEMINI_API_KEY != 'YOUR_GEMINI_API_KEY' else 'No / Placeholder'}")
    print(f"LangChain LLM Initialized: {'Yes (' + model_name_to_use + ')' if llm else 'No! Check API Key/Config!'}")
    print(f"LangChain Agent Executor Initialized: {'Yes' if agent_executor else 'No!'}")
    if not agent_executor or not llm:
        print("\n--- WARNING: LangChain components failed to initialize. The '/recommend' endpoint will fail. ---")
        print("--- Please check your GEMINI_API_KEY and potential network issues. ---")
    print("\nFlask server running. Open http://127.0.0.1:5000 or http://<your-ip>:5000 in your browser.")
    print("Press CTRL+C to quit.\n")
    # Change host to '0.0.0.0' to make it accessible on your network
    app.run(debug=True, host='0.0.0.0', port=5000) # Use debug=False in production
import os
import traceback

# --- PyTorch Model Definition (MUST match the training script) ---
# Define the model architecture again here so we can load the state_dict

class TransformerEncoderBlock(nn.Module):
    """PyTorch Transformer Encoder Block"""
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        # Ensure batch_first=True for consistency
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout_rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs) # Self-attention
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output) # Add & Norm
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output) # Add & Norm
        return out2

class StockRecommenderModel(nn.Module):
    """PyTorch Stock Recommender Model using Transformer"""
    # Use the same hyperparameters as used during training for instantiation
    def __init__(self, input_dim, num_transformer_blocks=4, embed_dim=128, num_heads=4, ff_dim=128, mlp_units=[64], dropout=0.15, mlp_dropout=0.25):
        super().__init__()
        # Assuming embed_dim = input_dim as potentially done in training
        # If a projection layer was used in training, it must be included here too.
        if embed_dim != input_dim:
            print(f"Instantiating model with embed_dim={input_dim} to match input features.")
            embed_dim = input_dim # Ensure consistency if no projection used

        # Stack of transformer blocks
        self.transformer_blocks = nn.Sequential(
            *[TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_transformer_blocks)]
        )

        # MLP Head for final output
        mlp_layers = []
        current_dim = embed_dim # Dimension after transformer blocks
        # Build MLP layers dynamically
        for units in mlp_units:
            mlp_layers.append(nn.Linear(current_dim, units))
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Dropout(mlp_dropout))
            current_dim = units
        # Final layer outputs 2 values
        mlp_layers.append(nn.Linear(current_dim, 2))

        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(self, x):
        # Pass through transformer blocks
        x = self.transformer_blocks(x)
        # Apply global average pooling across sequence length dimension (dim=1)
        x = torch.mean(x, dim=1)
        # Pass through MLP head
        x = self.mlp_head(x)
        return x

# --- Configuration ---
# File paths for loading saved artifacts
MODEL_PATH = 'stock_recommender_model_pytorch.pth' # PyTorch model file
SCALER_PATH = 'scaler.pkl' # Scaler object
INFO_PATH = 'stock_info.pkl' # Stock metadata dictionary
SEQ_PATH = 'last_sequences.pkl' # Last sequences for prediction dictionary

# --- Global Variables ---
# These will hold the loaded objects
model = None
scaler = None
stock_info = None
last_unscaled_sequence = None
artifacts_loaded = False # Flag to track if loading was successful
available_sectors = [] # List of sectors for the web form dropdown
# Determine device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_features = -1 # Placeholder for number of features, determined during loading

# --- Helper Functions ---

def load_artifacts():
    """Loads the saved PyTorch model state_dict, scaler, stock info, and sequences."""
    # Declare intent to modify global variables
    global model, scaler, stock_info, last_unscaled_sequence, artifacts_loaded, available_sectors, n_features, device
    print(f"Attempting to load artifacts for device: {device}...")
    # List required files
    required_files = [MODEL_PATH, SCALER_PATH, INFO_PATH, SEQ_PATH]
    # Check if all files exist
    if not all(os.path.exists(p) for p in required_files):
        missing = [p for p in required_files if not os.path.exists(p)]
        print(f"Error: Missing artifact files: {', '.join(missing)}")
        artifacts_loaded = False
        return False # Indicate failure

    try:
        # Load scaler first to potentially determine n_features
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        print(f"Scaler loaded from {SCALER_PATH}")
        # Try to get number of features from the scaler object
        if hasattr(scaler, 'n_features_in_'):
            n_features = scaler.n_features_in_
            print(f"Inferred n_features from scaler: {n_features}")
        else:
            # Fallback: Infer from the shape of the first sequence in the loaded data
            print("Warning: Cannot infer n_features from scaler. Trying sequence data...")
            with open(SEQ_PATH, 'rb') as f: temp_seq = pickle.load(f)
            if temp_seq: # Check if sequence dictionary is not empty
                first_key = next(iter(temp_seq)) # Get the first ticker key
                n_features = temp_seq[first_key].shape[1] # Get number of features (columns)
                print(f"Inferred n_features from sequence data: {n_features}")
            else:
                # Critical error if n_features cannot be determined
                print("Error: Cannot determine n_features for model instantiation from sequence data.")
                return False

        # Load stock info dictionary
        with open(INFO_PATH, 'rb') as f: stock_info = pickle.load(f)
        print(f"Stock info loaded from {INFO_PATH} ({len(stock_info)} tickers)")

        # Load last sequences dictionary
        with open(SEQ_PATH, 'rb') as f: last_unscaled_sequence = pickle.load(f)
        print(f"Last sequences loaded from {SEQ_PATH} ({len(last_unscaled_sequence)} tickers)")

        # --- Instantiate and Load PyTorch Model ---
        # Ensure n_features was successfully determined
        if n_features <= 0:
            print("Error: Failed to determine number of features required for model.")
            return False

        # Instantiate the model structure.
        # **Crucial**: Hyperparameters MUST match those used during training.
        model_instance = StockRecommenderModel(
            input_dim=n_features,
            # Ensure these match the values from train_recommender_pytorch.py
            num_transformer_blocks=4,
            embed_dim=128, # Or n_features if that was used in training
            num_heads=4,
            ff_dim=128,
            mlp_units=[64],
            dropout=0.15,
            mlp_dropout=0.25
        ).to(device) # Move the model structure to the target device

        # Load the learned parameters (state dictionary) from the file
        # map_location=device ensures correct loading across different devices (e.g., train on GPU, load on CPU)
        model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=device))

        # Set the model to evaluation mode (disables dropout, etc.)
        model_instance.eval()
        model = model_instance # Assign the loaded model to the global variable
        print(f"PyTorch model loaded successfully from {MODEL_PATH} and set to eval mode.")
        # -----------------------------------------

        # Populate the list of available sectors for the web form
        if stock_info:
            sectors = set(v.get('sector', 'Unknown') for v in stock_info.values())
            available_sectors = sorted([s for s in sectors if s != 'Unknown'])
            print(f"Available sectors found: {available_sectors}")
        else:
            available_sectors = []

        # Mark artifacts as successfully loaded
        artifacts_loaded = True
        print("All artifacts loaded successfully.")
        return True # Indicate success
    except FileNotFoundError as e:
        print(f"Error: Artifact file not found - {e}")
        artifacts_loaded = False
        return False
    except Exception as e:
        # Catch any other exceptions during loading
        print(f"Error loading artifacts: {e}")
        traceback.print_exc() # Print detailed error traceback
        artifacts_loaded = False
        return False

def _get_stock_description(ticker, stock_info_dict):
    """Generates a brief description for a given stock ticker."""
    # Specific descriptions for known tickers
    if ticker == 'AAPL': return "Apple: Leading tech, strong brand loyalty (smartphones, services)."
    if ticker == 'MSFT': return "Microsoft: Tech leader in cloud (Azure), software (Office 365)."
    if ticker == 'NVDA': return "NVIDIA: Leader in GPUs for gaming, AI, data centers."
    # Generic description using loaded stock info
    if ticker in stock_info_dict:
        name = stock_info_dict[ticker].get('name', ticker)
        sector = stock_info_dict[ticker].get('sector', 'Unknown sector')
        return f"{name}: Operates in the {sector} sector."
    else:
        # Fallback if info is missing
        return f"{ticker}: Stock listed on the market."

def generate_recommendations(investment_goal, risk_tolerance, preferred_sector,
                            time_horizon, esg_importance, top_n=3):
    """Generates stock recommendations based on user preferences and the loaded PyTorch model."""
    # Access global variables holding loaded artifacts
    global model, scaler, stock_info, last_unscaled_sequence, artifacts_loaded, device

    # Check if artifacts are loaded before proceeding
    if not artifacts_loaded:
        print("Error attempting to generate recommendations: Artifacts not loaded.")
        raise RuntimeError("Model artifacts not loaded. Cannot generate recommendations.")

    # --- Map User Preferences to Scores ---
    risk_map = {'Conservative': 0.3, 'Moderate': 0.6, 'Aggressive': 1.0}
    time_map = {'Short Term (< 1 year)': 0, 'Medium Term (1-5 years)': 0.5, 'Long Term (> 5 years)': 1.0}
    risk_score = risk_map.get(risk_tolerance, 0.6) # Default to Moderate if key not found
    time_score = time_map.get(time_horizon, 0.5) # Default to Medium Term
    # Convert ESG importance to a 0-1 weight
    try:
        esg_input = float(esg_importance)
        esg_weight = min(max(esg_input / 100 if esg_input > 1 else esg_input, 0), 1)
    except (ValueError, TypeError):
        esg_weight = 0.5 # Default if conversion fails

    # --- Initialize Scoring Dictionaries ---
    predictions = {} # Stores raw weighted return predictions
    match_scores = {} # Stores final preference match scores

    # Get list of tickers for which we have both sequence data and stock info
    valid_tickers = [t for t in last_unscaled_sequence.keys() if t in stock_info]
    print(f"Evaluating {len(valid_tickers)} potential stocks...")

    # Return empty list if no tickers can be evaluated
    if not valid_tickers: return []

    # --- Inference Loop ---
    # Ensure model is in evaluation mode
    model.eval()
    # Disable gradient calculation for efficiency during inference
    with torch.no_grad():
        # Iterate through each valid ticker
        for ticker in valid_tickers:
            stock_details = stock_info[ticker] # Get pre-loaded info for the ticker

            # --- Calculate Preference Match Scores (same logic as TF version) ---
            # Sector Match
            sector_match = 1.0 if stock_details.get('sector') == preferred_sector else (0.5 if preferred_sector == 'Any' else 0.3)
            # Risk Match (using Beta)
            beta = stock_details.get('beta', 1.0) or 1.0 # Default beta = 1.0
            target_beta_low, target_beta_high = (0, 0.8) if risk_score <= 0.4 else ((0.8, 1.2) if risk_score < 0.7 else (1.2, 10))
            if target_beta_low <= beta <= target_beta_high: risk_match = 1.0
            else:
                mid_target_beta = max(0.1, (target_beta_low + target_beta_high) / 2)
                risk_match = max(0, 1.0 - abs(beta - mid_target_beta) / mid_target_beta)
            # ESG Match
            esg_score_val = stock_details.get('esg_score', 50) or 50 # Default ESG = 50
            esg_match_raw = esg_score_val / 100.0
            esg_match = (esg_match_raw * esg_weight) + (1 * (1-esg_weight))

            # --- Prepare Input for PyTorch Model ---
            unscaled_sequence_np = last_unscaled_sequence[ticker] # Get the unscaled sequence (numpy)
            n_steps, n_features_seq = unscaled_sequence_np.shape

            # Scale the sequence using the loaded scaler
            try:
                # Reshape to 2D, scale, reshape back to 3D (with batch size 1)
                sequence_scaled_np = scaler.transform(unscaled_sequence_np.reshape(-1, n_features_seq)).reshape(1, n_steps, n_features_seq)
            except ValueError as e:
                print(f"Warning: Skipping {ticker}: Scaling error - {e}")
                continue # Skip if scaling fails (e.g., feature mismatch)
            except Exception as e:
                print(f"Warning: Skipping {ticker}: Unexpected scaling error - {e}")
                traceback.print_exc()
                continue

            # Convert the scaled numpy array to a PyTorch tensor
            sequence_tensor = torch.tensor(sequence_scaled_np, dtype=torch.float32).to(device)

            # --- Get Model Prediction ---
            try:
                # Pass the tensor through the model
                output_tensor = model(sequence_tensor)
                # Move the output tensor to CPU, convert to numpy array, get the first (only) batch result
                prediction = output_tensor.cpu().numpy()[0] # Shape (2,)
            except Exception as e:
                print(f"Error predicting for {ticker}: {e}")
                traceback.print_exc()
                continue # Skip ticker if prediction fails

            # --- Calculate Overall Match Score (same logic as TF version) ---
            # Combine 1M and 3M predictions based on time horizon
            weighted_return = (1 - time_score) * prediction[0] + time_score * prediction[1]
            # Calculate dividend score
            dividend_yield = stock_details.get('dividend_yield', 0) or 0
            dividend_score = min(dividend_yield / 0.05, 1.0) if dividend_yield else 0.0
            # Define weights based on investment goal
            w_return, w_dividend, w_sector, w_risk, w_esg = 0, 0, 0, 0, 0
            if investment_goal == "Growth - Focus on capital appreciation": w_return, w_dividend, w_sector, w_risk, w_esg = 0.5, 0.0, 0.2, 0.2, 0.1
            elif investment_goal == "Income - Focus on dividends": w_return, w_dividend, w_sector, w_risk, w_esg = 0.1, 0.5, 0.1, 0.2, 0.1
            else: w_return, w_dividend, w_sector, w_risk, w_esg = 0.3, 0.3, 0.15, 0.15, 0.1 # Balanced
            # Calculate final score
            scaled_return_contrib = max(-0.5, weighted_return) * w_return # Limit negative impact
            match_score = (scaled_return_contrib + dividend_score * w_dividend + sector_match * w_sector +
                           risk_match * w_risk + esg_match * w_esg)
            match_score = max(0, match_score) # Ensure score >= 0

            # Store results (ensure float type for JSON compatibility later if needed)
            predictions[ticker] = float(weighted_return)
            match_scores[ticker] = float(match_score)

    # --- Select Top N Recommendations ---
    # Parse top_n input, default to 3 if invalid
    try: top_n_parsed = int(top_n)
    except (ValueError, TypeError): top_n_parsed = 3
    # Sort tickers by match score (descending) and take the top N
    top_tickers = sorted(match_scores.keys(), key=lambda x: match_scores[x], reverse=True)[:top_n_parsed]

    # --- Format Recommendations for Display ---
    recommendations_list = []
    for ticker in top_tickers:
        stock_details = stock_info[ticker]
        # Create a dictionary for each recommended stock
        recommendations_list.append({
            'ticker': ticker,
            'name': stock_details.get('name', ticker),
            'price': float(stock_details.get('latest_close', 0)),
            'pe_ratio': float(stock_details.get('pe_ratio', 0)),
            'dividend_yield': float((stock_details.get('dividend_yield', 0) or 0) * 100), # As percentage
            'match_percentage': int(min(match_scores[ticker] * 100, 100)), # Cap at 100
            'predicted_return_weighted': predictions[ticker],
            'sector': stock_details.get('sector', 'Unknown'),
            'description': _get_stock_description(ticker, stock_info)
        })
    print(f"Generated {len(recommendations_list)} recommendations.")
    return recommendations_list # Return the list of recommendation dictionaries

# --- Flask App Setup ---
app = Flask(__name__) # Initialize Flask app
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    """Route to display the main input form (index.html)."""
    # Check if artifacts loaded successfully before rendering
    if not artifacts_loaded:
        return "Error: Model artifacts are not loaded. Please check server logs.", 500
    # Render the HTML template, passing the list of available sectors
    return render_template('index.html', sectors=available_sectors)

@app.route('/recommend', methods=['POST'])
def recommend_route():
    """Route to handle form submission and display recommendations (results.html)."""
    # Check artifacts again
    if not artifacts_loaded:
        return "Error: Model artifacts are not loaded. Please check server logs.", 500

    try:
        # Retrieve data submitted via the HTML form
        goal = request.form.get('investment_goal')
        risk = request.form.get('risk_tolerance')
        sector = request.form.get('preferred_sector')
        horizon = request.form.get('time_horizon')
        esg = request.form.get('esg_importance', 0.5) # Default if not provided
        top_n = request.form.get('top_n', 3) # Default if not provided

        # Basic validation: check if required fields were received
        if not all([goal, risk, sector, horizon]):
            # Render the results page with an error message
            return render_template('results.html', error_message="Missing required form fields."), 400

        # Log the received data for debugging purposes
        print(f"Received Request Data: Goal={goal}, Risk={risk}, Sector={sector}, Horizon={horizon}, ESG={esg}, TopN={top_n}")

        # Call the function to generate recommendations using the form data
        recommendations = generate_recommendations(
            investment_goal=goal,
            risk_tolerance=risk,
            preferred_sector=sector,
            time_horizon=horizon,
            esg_importance=esg,
            top_n=top_n
        )

        # Render the results page, passing the generated list of recommendations
        return render_template('results.html', recommendations=recommendations)

    except Exception as e:
        # Catch any exceptions during recommendation generation
        print(f"Error processing /recommend request: {e}")
        traceback.print_exc() # Print detailed error traceback to console
        # Render the results page with a generic error message
        return render_template('results.html', error_message="An error occurred while generating recommendations."), 500

# --- Main Execution ---
# This block runs when the script is executed directly (python app_pytorch.py)
if __name__ == '__main__':
    # Attempt to load all necessary artifacts when the application starts
    load_artifacts()
    # Start the Flask development server
    # host='127.0.0.1' makes it accessible only from the local machine
    # debug=True enables auto-reloading and provides detailed error pages during development
    app.run(host='127.0.0.1', port=5000, debug=True)
