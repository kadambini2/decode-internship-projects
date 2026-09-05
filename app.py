import json
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional

# ------------------------------------------------------------------------------
# 1. Page Configuration & UI Setup
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="DecodeLabs - Project 1 Extraction Engine",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Project 1: Deterministic Data Extraction Engine")
st.caption("Powered by DecodeLabs Project 1 Blueprint | Zero-Variance Pipeline")

# Sidebar for API Key and Model Controls
with st.sidebar:
    st.header("API & Pipeline Settings")
    api_key = st.text_input("OpenAI API Key", type="password")
    selected_model = st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"])
    
    # Enforce hardcoded zero temperature visually for compliance
    temperature = st.slider("Temperature (Fixed)", min_value=0.0, max_value=0.0, value=0.0, step=0.0)
    st.info("Temperature fixed at 0.0 to enforce deterministic outputs.")

# ------------------------------------------------------------------------------
# 2. Production System Prompt Template (Static Prefix First for Caching)
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
# 3. Main Application Interface
# ------------------------------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Raw Unstructured Data Input")
    
    # Gatekeeper Test Case default text (purposefully omits phone number)
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
            st.error("Please enter your OpenAI API key in the sidebar.")
        elif not raw_input.strip():
            st.warning("Please provide input text to extract.")
        else:
            try:
                # Initialize API client
                client = OpenAI(api_key=api_key)
                
                # Format prompt
                formatted_prompt = SYSTEM_PROMPT.replace("{RAW_USER_DATA}", raw_input)
                
                with st.spinner("Processing through Compilation Engine..."):
                    # Call LLM with forced 0.0 temperature
                    response = client.chat.completions.create(
                        model=selected_model,
                        temperature=0.0,
                        messages=[{"role": "user", "content": formatted_prompt}]
                    )
                    
                    raw_response = response.choices[0].message.content.strip()
                    
                    # Clean up response if backticks were generated
                    if raw_response.startswith("```"):
                        raw_response = raw_response.strip("`").replace("json\n", "").strip()

                    # Step 1 Validation: Syntax Parsing
                    parsed_json = json.loads(raw_response)
                    
                    st.success("Extraction Completed Successfully!")
                    st.json(parsed_json)
                    
                    # Highlight Gatekeeper logic compliance
                    if parsed_json.get("contact_phone") is None:
                        st.info(" Gatekeeper Gate Passed: Missing phone number deterministically returned `null`.")
                        
            except json.JSONDecodeError as e:
                st.error(f"JSON Parsing Error: The engine output failed syntax parsing.\n\nError: {str(e)}")
            except Exception as e:
                st.error(f"API Error: {str(e)}")
