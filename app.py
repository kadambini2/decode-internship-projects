import json
import urllib.request
import urllib.error
import streamlit as st

# ------------------------------------------------------------------------------
# 1. Configuration & Interface Layout
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="DecodeLabs - Project 1 Extraction Engine",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Project 1: Deterministic Data Extraction Engine")
st.caption("Powered by DecodeLabs Project 1 Blueprint | Direct REST API Pipeline")

with st.sidebar:
    st.header("API & Pipeline Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    selected_model = st.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-pro"])
    
    # Hardcoded 0.0 temperature for Project 1 compliance
    temperature = st.slider("Temperature (Fixed)", min_value=0.0, max_value=0.0, value=0.0, step=0.0)
    st.info("Temperature hardcoded to 0.0 to eliminate token output variance.")

# ------------------------------------------------------------------------------
# 2. Production System Prompt Template
# ------------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a deterministic data extraction engine. Extract information from the provided raw text input into the exact JSON schema specified below.

Output Rules:
- Output plain text valid JSON only. Do not include markdown headers, triple backticks, or conversational filler.
- Map missing or omitted fields directly to null.
- Extract values exactly as present in raw text without hallucinating placeholder values.

JSON Schema Mold:
{
  "customer_name": "string or null",
  "order_number": "string or null",
  "complaint_type": "string or null",
  "severity_level": integer or null,
  "contact_phone": "string or null"
}

<example>
Input:
###
Subject: Broken order #99482
Name: Sarah Jenkins (Phone: 555-0199)
Message: I received a completely shattered vase today. Extremely angry about this service!
###

Output:
{
  "customer_name": "Sarah Jenkins",
  "order_number": "99482",
  "complaint_type": "Damaged Goods",
  "severity_level": 5,
  "contact_phone": "555-0199"
}
</example>

<example>
Input:
###
Hi team, my name is Alex Reed. I tried logging into my account for order ORD-11223 but got error code 500.
###

Output:
{
  "customer_name": "Alex Reed",
  "order_number": "ORD-11223",
  "complaint_type": "Login Issue",
  "severity_level": 3,
  "contact_phone": null
}
</example>

###
{RAW_USER_DATA}
###"""

# ------------------------------------------------------------------------------
# 3. Direct REST API Call Function (No External SDKs Required)
# ------------------------------------------------------------------------------
def call_gemini_api(api_key: str, model: str, prompt: str) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json"
        }
    }
    
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, 
        data=data, 
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        return result["candidates"][0]["content"]["parts"][0]["text"]

# ------------------------------------------------------------------------------
# 4. Application Processing Logic
# ------------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Raw Unstructured Data Input")
    
    default_text = """From: Marcus Vance
Subject: Delayed delivery of items

I placed order #884920 two weeks ago and haven't received any shipment update.
This is unacceptable service and I demand a fast response!"""

    raw_input = st.text_area(
        "Paste messy text or customer support email below:",
        value=default_text,
        height=220
    )
    
    extract_btn = st.button("Extract Structured JSON", type="primary", use_container_width=True)

with col2:
    st.subheader("Deterministic Output Matrix")
    
    if extract_btn:
        if not api_key:
            st.error("Please enter your Gemini API key in the sidebar.")
        elif not raw_input.strip():
            st.warning("Please provide input text to extract.")
        else:
            try:
                formatted_prompt = SYSTEM_PROMPT.replace("{RAW_USER_DATA}", raw_input)
                
                with st.spinner("Processing through Compilation Engine..."):
                    raw_response = call_gemini_api(api_key, selected_model, formatted_prompt).strip()
                    
                    # Clean markdown wrappers if present
                    if raw_response.startswith("```"):
                        raw_response = raw_response.strip("`").replace("json\n", "").strip()

                    parsed_json = json.loads(raw_response)
                    
                    st.success("Extraction Completed Successfully!")
                    st.json(parsed_json)
                    
                    if parsed_json.get("contact_phone") is None:
                        st.info("Gatekeeper Gate Passed: Missing phone number deterministically returned `null`.")
                        
            except urllib.error.HTTPError as e:
                error_details = e.read().decode("utf-8")
                st.error(f"API Request Failed ({e.code}): {error_details}")
            except json.JSONDecodeError as e:
                st.error(f"JSON Parsing Error: Engine output failed syntax parsing.\n\nError: {str(e)}")
            except Exception as e:
                st.error(f"Unexpected Error: {str(e)}")
