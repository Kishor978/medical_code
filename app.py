import streamlit as st
import asyncio
import nest_asyncio
from main import AgenticLoop, system_prompt, get_icd_10_parser_tool
from input_processing import extract_text_from_file, extract_text_from_image

nest_asyncio.apply()

# --- Configuration and Helpers ---

# Load custom CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

async def get_response(agent):
    """Generates a response from the agent."""
    try:
        response = await agent.generate_response()
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"❌ Error generating response: {e}")
        return f"Sorry, an error occurred while processing your request: {e}"

# Load custom CSS
load_css("style.css")

# --- Custom Header with Logo and Title Alignment ---
LOGO_PATH = "logo.png" # <--- IMPORTANT: Change this to your logo's filename and path if different!

# Use columns for logo and title
col1, col2 = st.columns([0.1, 0.9]) # Adjust column ratios as needed

with col1:
    try:
        st.image(LOGO_PATH, width=100, use_container_width=True, output_format="PNG") # Smaller width for inline logo
    except FileNotFoundError:
        st.error(f"Error: Logo file not found at {LOGO_PATH}. Please ensure it's in the correct directory.")

with col2:
    st.markdown('<h1 class="main-title-inline">AI Medical Coder</h1>', unsafe_allow_html=True) # New class for inline title

st.markdown(
    """
    <p class="subtitle">Intelligent ICD-10 Coding Assistant powered by AI</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# --- Session State Initialization ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "agent" not in st.session_state:
    st.session_state.agent = AgenticLoop(
        messages=st.session_state.chat_history.copy(),
        tools=[get_icd_10_parser_tool()],
        system_prompt=system_prompt
    )

if "file_ready" not in st.session_state:
    st.session_state.file_ready = False

if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

# --- Handle Post-Upload Processing ---
if st.session_state.file_ready:
    file_text = st.session_state.uploaded_text

    st.session_state.chat_history.append({
        "role": "user",
        "content": f"**Uploaded File Content:**\n\n```\n{file_text[:1000]}...\n```"
    })

    with st.expander("**Full Extracted Note (Click to view)**"):
        st.text(file_text)

    st.session_state.agent = AgenticLoop(
        messages=[{
            "role": "user",
            "content": f"Please analyze these patient notes and identify ICD-10 codes:\n\n{file_text}"
        }],
        tools=[get_icd_10_parser_tool()],
        system_prompt=system_prompt
    )

    with st.status("🔍 Analyzing extracted content...", expanded=True) as status:
        try:
            reply = asyncio.run(get_response(st.session_state.agent))
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            status.update(label="✅ Analysis complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"❌ Analysis failed: {e}", state="error", expanded=True)

    st.session_state.file_ready = False
    st.rerun()

# --- Example Prompt Hint ---
if not st.session_state.chat_history:
    st.info("💡 **Welcome!** You can upload a patient note or ask a medical coding question below. Our AI will assist you with ICD-10 codes.")

# --- Render Chat History ---
if st.session_state.chat_history:
    # --- Render Chat History (Flat) ---
    for msg in st.session_state.chat_history:
        avatar_icon = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar_icon):
            st.markdown(msg["content"], unsafe_allow_html=True)


# --- Chat Input with Integrated File Uploader ---
col_upload_btn, col_chat_input = st.columns([1, 10])

with col_upload_btn:
    with st.popover("📎", help="Upload a file"):
        st.markdown("**Upload Patient Note**")
        uploaded_file_in_popover = st.file_uploader(
            "Select file (.txt, .pdf, .docx, .jpg, .png)",
            type=["txt", "pdf", "docx", "jpg", "jpeg", "png"],
            key="popover_file_uploader",
            label_visibility="collapsed"
        )

        # File processing logic
        if uploaded_file_in_popover:
            file_text = None
            with st.spinner("Processing file..."):
                if uploaded_file_in_popover.name.endswith((".txt", ".pdf", ".docx")):
                    file_text = extract_text_from_file(uploaded_file_in_popover)
                elif uploaded_file_in_popover.name.endswith((".jpg", ".jpeg", ".png")):
                    file_text = extract_text_from_image(uploaded_file_in_popover)

            if file_text:
                st.session_state.uploaded_text = file_text
                st.session_state.file_ready = True
                st.rerun()  # 💡 Automatically rerun the app to trigger analysis
            else:
                st.error("❌ Failed to extract content from uploaded file.")

with col_chat_input:
    prompt = st.chat_input("Ask about a patient condition or ICD-10 code...", key="chat_input")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    st.session_state.agent = AgenticLoop(
        messages=st.session_state.chat_history.copy(),
        tools=[get_icd_10_parser_tool()],
        system_prompt=system_prompt
    )

    with st.spinner("🤖 AI is thinking..."):
        reply = asyncio.run(get_response(st.session_state.agent))
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
    st.rerun()