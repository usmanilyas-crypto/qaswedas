import os
import base64
from io import BytesIO
import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# Load API keys from your local .env file (ignored if the file doesn't exist)
load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Workflow Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI Workflow Assistant")
st.markdown("### Gemini → AI Reviewer → AI Finalizer")
st.write(
    "Enter a task below. Gemini will perform the initial work, "
    "an AI reviewer will check it, and the finalizer will prepare "
    "the final response."
)

# --------------------------------------------------
# CHECK API KEYS (.env locally, Streamlit secrets when deployed)
# --------------------------------------------------

def get_key(name: str):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


google_api_key = get_key("GOOGLE_API_KEY")
openrouter_api_key = get_key("OPENROUTER_API_KEY")

if not google_api_key:
    st.error("GOOGLE_API_KEY was not found. Please check your .env file or Streamlit secrets.")
    st.stop()

if not openrouter_api_key:
    st.error("OPENROUTER_API_KEY was not found. Please check your .env file or Streamlit secrets.")
    st.stop()

# Change the model name here (or in .env) if Google renames or retires it
GEMINI_MODEL = get_key("GEMINI_MODEL") or "gemini-3.6-flash"
# Use a specific chat model; "openrouter/free" can pick moderation models
OPENROUTER_MODEL = get_key("OPENROUTER_MODEL") or "MiniMax-M3.1-Flash (free tier"

# --------------------------------------------------
# CREATE AI MODELS
# --------------------------------------------------

@st.cache_resource
def create_models():
    gemini_model = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=google_api_key,
        temperature=0.2,
        max_retries=2
    )

    reviewer_model = ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.1,
        max_retries=2
    )

    finalizer_model = ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_api_key,
        temperature=0.2,
        max_retries=2
    )

    return gemini_model, reviewer_model, finalizer_model


def to_text(response) -> str:
    """LangChain returns a message; some Gemini models return a list of parts."""
    content = response.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content

# --------------------------------------------------
# FILE READING
# --------------------------------------------------

MAX_CHARS = 30000  # limit how much file text is sent to the AI


def extract_text(file) -> str:
    name = file.name.lower()
    if name.endswith((".xlsx", ".xls")):
        import pandas as pd
        sheets = pd.read_excel(file, sheet_name=None)
        return "\n\n".join(
            f"[Sheet: {sheet}]\n{df.to_csv(index=False)}" for sheet, df in sheets.items()
        )
    if name.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(file).to_csv(index=False)
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(file).pages)
    if name.endswith(".docx"):
        from docx import Document
        return "\n".join(p.text for p in Document(file).paragraphs)
    return file.read().decode("utf-8", errors="ignore")


images = []  # filled when the workflow runs
MAX_IMAGES = 6
MAX_PDF_PAGES_AS_IMAGES = 5


def pil_to_data_url(img) -> str:
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))  # keep uploads small
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def pdf_pages_to_images(data: bytes) -> list:
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(data)
    count = min(len(pdf), MAX_PDF_PAGES_AS_IMAGES)
    return [pil_to_data_url(pdf[i].render(scale=1.5).to_pil()) for i in range(count)]


def ask_model(model, prompt: str):
    """Send text, plus any uploaded images, to a model."""
    if images:
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": u}} for u in images
        ]
        return model.invoke([HumanMessage(content=content)])
    return model.invoke(prompt)


# --------------------------------------------------
# TASK INPUT
# --------------------------------------------------

st.subheader("Enter Your Task")

task = st.text_area(
    "What would you like the AI workflow to do?",
    height=180,
    placeholder=(
        "Example: Review these accounting journal entries, "
        "identify errors, explain the corrections, and provide "
        "the corrected entries."
    )
)

uploaded_files = st.file_uploader(
    "Optional: upload files (Excel, CSV, PDF, Word, text, images)",
    type=["xlsx", "xls", "csv", "pdf", "docx", "txt", "png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True
)

run_button = st.button(
    "Run AI Workflow",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# RUN THE THREE-STAGE WORKFLOW
# --------------------------------------------------

if run_button:

    if not task.strip():
        st.warning("Please enter a task before running the workflow.")
        st.stop()

    # Read uploaded files: text goes into the task, images are attached separately
    images.clear()
    if uploaded_files:
        from PIL import Image
        parts = []
        for f in uploaded_files:
            fname = f.name.lower()
            try:
                if fname.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    images.append(pil_to_data_url(Image.open(f)))
                    parts.append(f"=== IMAGE: {f.name} (attached to this message) ===")
                elif fname.endswith(".pdf"):
                    text = extract_text(f)
                    if len(text.strip()) < 50:  # scanned PDF: read pages as images
                        images.extend(pdf_pages_to_images(f.getvalue()))
                        parts.append(f"=== PDF: {f.name} (scanned; first pages attached as images) ===")
                    else:
                        parts.append(f"=== FILE: {f.name} ===\n{text}")
                else:
                    parts.append(f"=== FILE: {f.name} ===\n{extract_text(f)}")
            except Exception as e:
                st.error(f"Could not read {f.name}: {e}")
                st.stop()
        if len(images) > MAX_IMAGES:
            st.warning(f"Only the first {MAX_IMAGES} images/pages were used.")
            del images[MAX_IMAGES:]
        file_text = "\n\n".join(parts)
        if len(file_text) > MAX_CHARS:
            st.warning(f"Files are large, so only the first {MAX_CHARS:,} characters were used.")
            file_text = file_text[:MAX_CHARS]
        task = f"{task}\n\nUse the following uploaded file contents:\n\n{file_text}"

    try:
        gemini_model, reviewer_model, finalizer_model = create_models()

        # STAGE 1: GEMINI
        with st.status("Stage 1: Gemini is working on your task...", expanded=False) as status:
            gemini_result = to_text(ask_model(gemini_model, 
                "You are a careful assistant. Complete the following task "
                f"thoroughly and accurately:\n\n{task}"
            ))
            status.update(label="Stage 1 complete: Gemini finished", state="complete")

        # STAGE 2: REVIEWER
        with st.status("Stage 2: Reviewer is checking the work...", expanded=False) as status:
            review_result = to_text(ask_model(reviewer_model, 
                "You are a strict reviewer. Check the response below against "
                "the original task. List any errors, omissions or unclear "
                "points, and suggest specific fixes.\n\n"
                f"ORIGINAL TASK:\n{task}\n\n"
                f"RESPONSE TO REVIEW:\n{gemini_result}"
            ))
            status.update(label="Stage 2 complete: Review finished", state="complete")

        # STAGE 3: FINALIZER
        with st.status("Stage 3: Finalizer is preparing the final answer...", expanded=False) as status:
            final_result = to_text(ask_model(finalizer_model, 
                "You are a finalizer. Using the original task, the draft, and "
                "the reviewer's feedback, write the best final answer. Apply "
                "the valid corrections and return only the final answer.\n\n"
                f"ORIGINAL TASK:\n{task}\n\n"
                f"DRAFT:\n{gemini_result}\n\n"
                f"REVIEW FEEDBACK:\n{review_result}"
            ))
            status.update(label="Stage 3 complete: Final answer ready", state="complete")

        # Fall back to Gemini's draft if the final answer is clearly not a real answer
        if len(final_result.strip()) < 40 or final_result.strip().lower().startswith("user safety"):
            st.warning("The finalizer returned an unusable answer, so Gemini's draft is shown instead.")
            final_result = gemini_result

        # RESULTS
        st.success("Workflow complete!")

        st.subheader("✅ Final Answer")
        st.markdown(final_result)

        with st.expander("Stage 1: Gemini's initial response"):
            st.markdown(gemini_result)

        with st.expander("Stage 2: Reviewer's feedback"):
            st.markdown(review_result)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
