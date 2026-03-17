"""
JobOps — AI-powered job search scoring and resume tailoring.
Single-file version. Run with: streamlit run app.py
"""

import streamlit as st
import requests
import sqlite3
import re
import io
from datetime import datetime
from openai import OpenAI


# ╔══════════════════════════════════════════════════════════════╗
# ║  PAGE CONFIG — MUST be the very first Streamlit command     ║
# ╚══════════════════════════════════════════════════════════════╝

st.set_page_config(
    page_title="JobOps",
    page_icon="💼",
    layout="wide",
)


# ╔══════════════════════════════════════════════════════════════╗
# ║  LLM PROVIDER CONFIGURATION                                 ║
# ╚══════════════════════════════════════════════════════════════╝

PROVIDERS = {
    "OpenAI": {
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "default": "gpt-4o-mini",
        "base_url": None,
        "key_name": "OPENAI_API_KEY",
    },
    "xAI (Grok)": {
        "models": ["grok-3-mini-beta", "grok-3-beta", "grok-2"],
        "default": "grok-3-mini-beta",
        "base_url": "https://api.x.ai/v1",
        "key_name": "XAI_API_KEY",
    },
    "DeepSeek": {
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "key_name": "DEEPSEEK_API_KEY",
    },
}

COUNTRY_CODES = {
    "us": "us", "usa": "us", "united states": "us",
    "uk": "gb", "gb": "gb", "united kingdom": "gb",
    "canada": "ca", "ca": "ca",
    "australia": "au", "au": "au",
    "germany": "de", "de": "de",
    "france": "fr", "fr": "fr",
    "india": "in", "in": "in",
    "netherlands": "nl", "nl": "nl",
    "brazil": "br", "br": "br",
    "singapore": "sg", "sg": "sg",
}

STATUS_OPTIONS = [
    "New", "Applied", "Interviewing", "Offer", "Rejected", "Withdrawn"
]


# ╔══════════════════════════════════════════════════════════════╗
# ║  FILE PARSING (TXT, PDF, DOCX, DOC)                         ║
# ╚══════════════════════════════════════════════════════════════╝

def parse_uploaded_file(uploaded_file) -> tuple[str, str | None]:
    """
    Parse an uploaded file and return (text_content, error_or_None).
    Supports .txt, .pdf, and .docx formats.
    Detects .doc and .odt with helpful conversion instructions.
    """
    filename = uploaded_file.name.lower()

    try:
        # ── Plain text ──
        if filename.endswith(".txt"):
            return uploaded_file.read().decode("utf-8"), None

        # ── PDF ──
        elif filename.endswith(".pdf"):
            import pdfplumber

            text_parts = []
            with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)

            if not text_parts:
                return "", (
                    "Could not extract text from this PDF. It may be "
                    "image-based (scanned). Try:\n"
                    "1. Open the PDF and Select All → Copy → paste into "
                    "the text box above, OR\n"
                    "2. Use a .docx or .txt version instead"
                )

            return "\n\n".join(text_parts), None

        # ── Word .docx (2007+) ──
        elif filename.endswith(".docx"):
            from docx import Document

            doc = Document(io.BytesIO(uploaded_file.read()))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            if not paragraphs:
                return "", "Word document appears to be empty."

            return "\n\n".join(paragraphs), None

        # ── Word .doc (97-2003) — not supported directly ──
        elif filename.endswith(".doc"):
            return "", (
                "**.doc files (Word 97-2003 format) are not supported.** "
                "The newer **.docx** format is required.\n\n"
                "**Easy fixes:**\n"
                "1. **Re-save as .docx**: Open the file in Word → "
                "File → Save As → choose \"Word Document (.docx)\"\n"
                "2. **Re-save as .pdf**: Open in Word → File → "
                "Save As → choose \"PDF\"\n"
                "3. **Copy & paste**: Open the file → Select All → "
                "Copy → paste into the text box above\n\n"
                "*Google Docs can also open .doc files and export as .docx*"
            )

        # ── LibreOffice .odt ──
        elif filename.endswith(".odt"):
            return "", (
                "**.odt files (LibreOffice) are not supported directly.**\n\n"
                "**Easy fixes:**\n"
                "1. Open in LibreOffice → File → Save As → "
                "choose \"Word (.docx)\" or \"PDF\"\n"
                "2. Copy & paste the text into the text box above"
            )

        # ── Anything else ──
        else:
            ext = filename.rsplit(".", 1)[-1] if "." in filename else "unknown"
            return "", (
                f"**.{ext}** files are not supported. "
                f"Please use **.txt**, **.pdf**, or **.docx**"
            )

    except ImportError as e:
        missing = "pdfplumber" if "pdfplumber" in str(e) else "python-docx"
        return "", (
            f"Missing library: **{missing}**. "
            f"Install it with: `pip install {missing}`"
        )
    except Exception as e:
        return "", f"Error reading file: {e}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  LLM CLIENT HELPERS                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def get_api_key(key_name: str) -> str:
    """Retrieve an API key from Streamlit secrets safely."""
    try:
        return st.secrets.get(key_name, "")
    except Exception:
        return ""


def get_client(provider_name: str) -> OpenAI:
    """Return an OpenAI-compatible client for the chosen provider."""
    cfg = PROVIDERS[provider_name]
    api_key = get_api_key(cfg["key_name"])
    if not api_key:
        raise ValueError(
            f"API key '{cfg['key_name']}' not found. "
            f"Add it to .streamlit/secrets.toml"
        )
    kwargs = {"api_key": api_key}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    return OpenAI(**kwargs)


def safe_get_client(provider_name: str):
    """Get LLM client with UI error handling."""
    try:
        return get_client(provider_name)
    except ValueError as e:
        st.error(str(e))
        st.stop()


# ╔══════════════════════════════════════════════════════════════╗
# ║  DATABASE (SQLite)                                           ║
# ╚══════════════════════════════════════════════════════════════╝

DB_FILE = "jobops.db"


def _db_connect():
    """Return a connection with Row factory enabled."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def db_init():
    """Create the jobs table if it doesn't exist."""
    with _db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                title             TEXT NOT NULL,
                company           TEXT DEFAULT '',
                location          TEXT DEFAULT '',
                description       TEXT DEFAULT '',
                url               TEXT DEFAULT '',
                salary            TEXT DEFAULT '',
                source            TEXT DEFAULT '',
                score             INTEGER DEFAULT 0,
                score_reason      TEXT DEFAULT '',
                tailored_summary  TEXT DEFAULT '',
                tailored_bullets  TEXT DEFAULT '',
                status            TEXT DEFAULT 'New',
                notes             TEXT DEFAULT '',
                created_at        TEXT,
                updated_at        TEXT
            )
        """)


def db_save_job(job: dict, score: int = 0, reason: str = "") -> int:
    """Insert a job and return its new ID."""
    now = datetime.now().isoformat()
    with _db_connect() as conn:
        cur = conn.execute(
            """INSERT INTO jobs
               (title, company, location, description, url, salary, source,
                score, score_reason, status, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.get("title", ""),
                job.get("company", ""),
                job.get("location", ""),
                job.get("description", ""),
                job.get("url", ""),
                job.get("salary", ""),
                job.get("source", ""),
                score,
                reason,
                "New",
                now,
                now,
            ),
        )
        return cur.lastrowid


def db_get_all_jobs() -> list:
    """Return all saved jobs, highest score first."""
    with _db_connect() as conn:
        rows = conn.execute(
            "SELECT * FROM jobs ORDER BY score DESC, created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def db_get_job(job_id: int) -> dict | None:
    """Return a single job by ID, or None."""
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return dict(row) if row else None


def db_update_status(job_id: int, status: str):
    """Update application status for a job."""
    with _db_connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = ?, updated_at = ? WHERE id = ?",
            (status, datetime.now().isoformat(), job_id),
        )


def db_update_tailored(job_id: int, summary: str, bullets: str):
    """Save tailored resume content for a job."""
    with _db_connect() as conn:
        conn.execute(
            "UPDATE jobs SET tailored_summary = ?, tailored_bullets = ?, "
            "updated_at = ? WHERE id = ?",
            (summary, bullets, datetime.now().isoformat(), job_id),
        )


def db_update_notes(job_id: int, notes: str):
    """Update freeform notes for a job."""
    with _db_connect() as conn:
        conn.execute(
            "UPDATE jobs SET notes = ?, updated_at = ? WHERE id = ?",
            (notes, datetime.now().isoformat(), job_id),
        )


def db_delete_job(job_id: int):
    """Permanently delete a job."""
    with _db_connect() as conn:
        conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))


def db_job_exists(title: str, company: str, url: str) -> bool:
    """Check if a job with same title+company+url is already saved."""
    with _db_connect() as conn:
        row = conn.execute(
            "SELECT id FROM jobs WHERE title = ? AND company = ? AND url = ?",
            (title, company, url),
        ).fetchone()
        return row is not None


# ╔══════════════════════════════════════════════════════════════╗
# ║  JOB SEARCH (Adzuna API — optional)                          ║
# ╚══════════════════════════════════════════════════════════════╝

def search_adzuna(
    query: str,
    country: str = "us",
    max_results: int = 15,
) -> tuple[list, str | None]:
    """
    Search via Adzuna API (free tier, 250 requests/month).
    Returns (list_of_job_dicts, error_string_or_None).
    """
    app_id = get_api_key("ADZUNA_APP_ID")
    app_key = get_api_key("ADZUNA_APP_KEY")

    if not app_id or not app_key:
        return [], (
            "Adzuna not configured. This is **optional** — you can use the "
            "**📋 Paste Job** tab without it.\n\n"
            "To enable search: get free credentials at "
            "[developer.adzuna.com](https://developer.adzuna.com) "
            "and add `ADZUNA_APP_ID` / `ADZUNA_APP_KEY` to "
            "`.streamlit/secrets.toml`"
        )

    cc = COUNTRY_CODES.get(country.lower().strip(), country.lower().strip()[:2])

    url = f"https://api.adzuna.com/v1/api/jobs/{cc}/search/1"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "results_per_page": min(max_results, 50),
        "what": query,
        "content-type": "application/json",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            return [], f"Adzuna returned HTTP {resp.status_code}: {resp.text[:200]}"

        data = resp.json()
        jobs = []

        for r in data.get("results", []):
            salary = ""
            sal_min = r.get("salary_min")
            sal_max = r.get("salary_max")
            if sal_min or sal_max:
                lo = f"${sal_min:,.0f}" if sal_min else "?"
                hi = f"${sal_max:,.0f}" if sal_max else "?"
                salary = f"{lo} – {hi}"

            jobs.append({
                "title": r.get("title", "N/A"),
                "company": r.get("company", {}).get("display_name", "N/A"),
                "location": r.get("location", {}).get("display_name", "N/A"),
                "description": r.get("description", "N/A"),
                "url": r.get("redirect_url", "#"),
                "salary": salary,
                "source": "Adzuna",
            })

        return jobs, None

    except requests.exceptions.Timeout:
        return [], "Request timed out. Try again."
    except requests.exceptions.ConnectionError:
        return [], "Connection failed. Check your internet."
    except Exception as e:
        return [], f"Unexpected error: {e}"


def parse_manual_job(
    title: str,
    company: str,
    description: str,
    url: str = "",
    location: str = "",
) -> dict:
    """Package a manually entered job into standard dict format."""
    return {
        "title": title.strip(),
        "company": company.strip(),
        "location": location.strip(),
        "description": description.strip(),
        "url": url.strip() or "#",
        "salary": "",
        "source": "Manual",
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║  AI OPERATIONS (Scoring & Tailoring)                         ║
# ╚══════════════════════════════════════════════════════════════╝

def ai_score_job(
    client: OpenAI,
    model: str,
    resume: str,
    job_desc: str,
) -> tuple[int, str]:
    """
    Score candidate-job fit from 0-100 with a one-line explanation.
    Returns (score: int, reason: str).
    """
    prompt = f"""You are a career advisor. Score how well this candidate fits the job.

CANDIDATE PROFILE:
{resume}

JOB DESCRIPTION:
{job_desc}

Respond in EXACTLY this format (two lines, nothing else):
SCORE: <integer 0-100>
REASON: <one sentence explaining the score>"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()

        m = re.search(r"SCORE:\s*(\d+)", text)
        score = min(100, max(0, int(m.group(1)))) if m else 0

        m2 = re.search(r"REASON:\s*(.+)", text, re.DOTALL)
        reason = m2.group(1).strip() if m2 else text

        return score, reason

    except Exception as e:
        return 0, f"Scoring error: {e}"


def ai_tailor_summary(
    client: OpenAI,
    model: str,
    resume: str,
    job_desc: str,
) -> str:
    """Generate a 4-6 sentence professional summary tailored to the job."""
    prompt = f"""You are an expert resume writer.

CANDIDATE PROFILE:
{resume}

TARGET JOB:
{job_desc}

Write a 4-6 sentence professional summary for the top of the candidate's
resume, tailored specifically to this job. Highlight the most relevant skills
and experience. Weave in keywords from the job description naturally.
Do NOT fabricate experience the candidate doesn't have.
Return only the summary text, no labels or headers."""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating summary: {e}"


def ai_tailor_bullets(
    client: OpenAI,
    model: str,
    resume: str,
    job_desc: str,
) -> str:
    """Generate 5-8 tailored resume bullet points."""
    prompt = f"""You are an expert resume writer.

CANDIDATE PROFILE:
{resume}

TARGET JOB:
{job_desc}

Generate 5-8 resume bullet points highlighting the candidate's most relevant
experience for this specific job. Follow these rules:
- Start each bullet with a strong action verb
- Quantify achievements where possible
- Use keywords from the job description naturally
- Do NOT fabricate experience the candidate doesn't have
- Return one bullet per line, each starting with •"""

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating bullets: {e}"


# ╔══════════════════════════════════════════════════════════════╗
# ║  PASSWORD PROTECTION                                         ║
# ╚══════════════════════════════════════════════════════════════╝

def check_password() -> bool:
    """Gate the app behind a simple password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        st.text_input(
            "🔒 Enter password to continue",
            type="password",
            on_change=password_entered,
            key="password",
        )
        return False

    if not st.session_state["password_correct"]:
        st.text_input(
            "🔒 Enter password to continue",
            type="password",
            on_change=password_entered,
            key="password",
        )
        st.error("😕 Incorrect password")
        return False

    return True


if not check_password():
    st.stop()


# ╔══════════════════════════════════════════════════════════════╗
# ║  INITIALIZE DATABASE                                         ║
# ╚══════════════════════════════════════════════════════════════╝

db_init()


# ╔══════════════════════════════════════════════════════════════╗
# ║  SESSION STATE DEFAULTS                                      ║
# ╚══════════════════════════════════════════════════════════════╝

if "resume" not in st.session_state:
    st.session_state.resume = ""
if "search_results" not in st.session_state:
    st.session_state.search_results = []
if "manual_score" not in st.session_state:
    st.session_state.manual_score = None


# ╔══════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — Settings & Resume Input                           ║
# ╚══════════════════════════════════════════════════════════════╝

with st.sidebar:
    st.header("⚙️ Settings")

    # LLM provider selection
    provider = st.selectbox("LLM Provider", list(PROVIDERS.keys()))
    cfg = PROVIDERS[provider]
    model = st.selectbox(
        "Model",
        cfg["models"],
        index=cfg["models"].index(cfg["default"]),
    )

    # Show key status
    key_present = bool(get_api_key(cfg["key_name"]))
    if key_present:
        st.success(f"✅ {cfg['key_name']} found")
    else:
        st.error(f"❌ {cfg['key_name']} missing")

    # Adzuna key status
    adzuna_ok = bool(get_api_key("ADZUNA_APP_ID") and get_api_key("ADZUNA_APP_KEY"))
    if adzuna_ok:
        st.success("✅ Adzuna keys found")
    else:
        st.caption("ℹ️ Adzuna not configured (optional — for API Search tab)")

    st.divider()

    # Resume / profile input
    st.subheader("📄 Your Resume / Profile")

    resume_input = st.text_area(
        "Paste your resume text",
        value=st.session_state.resume,
        height=200,
        placeholder="Skills, experience, education, certifications…",
    )
    if resume_input != st.session_state.resume:
        st.session_state.resume = resume_input

    uploaded = st.file_uploader(
        "Or upload a file",
        type=["txt", "pdf", "docx", "doc", "odt"],
        key="resume_upload",
        help="Supported: .txt, .pdf, .docx  •  .doc and .odt will show conversion instructions",
    )
    if uploaded:
        text, err = parse_uploaded_file(uploaded)
        if err:
            st.error(err)
        elif text:
            st.session_state.resume = text
            st.success(f"✅ Loaded {len(text.split())} words from {uploaded.name}")
            st.rerun()

    if st.session_state.resume:
        word_count = len(st.session_state.resume.split())
        st.caption(f"📝 {word_count} words loaded")
    else:
        st.caption("⚠️ No resume loaded — paste or upload above")

    # Clear resume button
    if st.session_state.resume:
        if st.button("🗑️ Clear Resume", key="btn_clear_resume"):
            st.session_state.resume = ""
            st.rerun()


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN TITLE & HOW-TO GUIDE                                  ║
# ╚══════════════════════════════════════════════════════════════╝

st.title("💼 JobOps — AI Job Scorer & Resume Tailorer")
st.caption(
    f"Using **{provider}** / `{model}`  •  "
    f"Resume: {'✅ loaded' if st.session_state.resume else '❌ not loaded'}"
)

# ── Collapsible How-To Guide ──
with st.expander("📖 How to Use This Application", expanded=False):
    st.markdown("""
    ### Quick Start (3 steps)

    1. **Load your resume** → Sidebar (left panel): paste text or upload a file (.txt, .pdf, .docx)
    2. **Score a job** → Use the **📋 Paste Job** tab: paste any job description and get an AI fit score
    3. **Tailor your resume** → Save scored jobs, then use the **✂️ Tailor** tab to generate custom content

    ---

    ### How to Copy Job Postings

    The **📋 Paste Job** tab works with any job posting from any site.
    Here's how to grab the info you need:

    #### LinkedIn
    1. Open the job listing on LinkedIn
    2. You'll see the job title and company name at the top — type these into the app
    3. Click **"Show more"** (or **"See more"**) to expand the full job description
    4. Select all the description text (from "About the role" or similar down to the bottom)
    5. **Ctrl+C** (or **Cmd+C** on Mac) to copy
    6. Paste into the **"Full Job Description"** box in the app
    7. *(Optional)* Copy the URL from your browser's address bar into the **"Job URL"** field

    > **💡 LinkedIn tip:** The URL in your browser bar works fine.
    > It usually looks like `linkedin.com/jobs/view/1234567890`

    #### Indeed
    1. Open the full job listing (click the job title from search results)
    2. The title, company, and location are at the top — type these into the app
    3. Scroll to the **"Full Job Description"** section
    4. Select all the text from the description area
    5. Copy and paste into the app
    6. *(Optional)* Copy the URL from your browser bar

    > **💡 Indeed tip:** Some listings have a "Show full description" link — click it first
    > to reveal all the content before copying.

    #### Glassdoor
    1. Open the job listing
    2. Click **"Show more"** if the description is truncated
    3. Copy the full description text
    4. Note: Glassdoor may require you to be logged in to see full descriptions

    #### Company Career Pages
    1. These are often the best source — full descriptions with no truncation
    2. Open the listing, select all the job description text, copy and paste
    3. Save the URL — it's the most direct link to the posting

    #### General Tips for Any Site
    - **Copy as much as possible** — the AI does better with the full description
      including requirements, qualifications, responsibilities, and benefits
    - **Don't worry about formatting** — bullet points, weird spacing, and
      extra characters are all fine. The AI handles messy text well
    - **Include "nice to have" sections** — these affect your score and help
      the AI tailor your resume more accurately
    - **Skip the boilerplate** — you can leave out equal opportunity statements,
      company mission paragraphs, etc., but it won't hurt if you include them

    ---

    ### What to Paste Where

    | App Field | What to Put |
    |---|---|
    | **Job Title** | Exact title from the listing (e.g., "Senior Python Developer") |
    | **Company** | Company name |
    | **Location** | City, state, "Remote", "Hybrid", etc. — whatever the listing says |
    | **Job URL** | *(Optional)* The URL from your browser — useful for tracking |
    | **Full Job Description** | Everything else: responsibilities, requirements, qualifications, skills, benefits |

    ---

    ### Tab Guide

    | Tab | What to do |
    |---|---|
    | **📋 Paste Job** | Found a job anywhere? Paste title + description → get instant AI fit score → save it |
    | **🔍 API Search** | *(Optional — requires free Adzuna keys)* Search by keywords + country → auto-score results |
    | **📊 Search Results** | Review scored API search results → save the good ones |
    | **💾 Saved Jobs** | Your persistent job tracker — update status, add notes, view tailored content |
    | **✂️ Tailor Resume** | Pick any saved job → generate tailored summary + bullet points → edit → save |

    ---

    ### Scoring Guide

    | Color | Score | Meaning |
    |---|---|---|
    | 🟢 | 70–100 | Strong match — worth applying |
    | 🟡 | 40–69 | Partial match — review carefully |
    | 🔴 | 0–39 | Weak match — likely not a fit |

    ---

    ### Typical Workflow

    1. Browse jobs on LinkedIn, Indeed, Glassdoor, company career pages, etc.
    2. When you find an interesting listing, copy the full job description
    3. Paste into the **📋 Paste Job** tab → fill in title, company → score it
    4. If the score is 🟢 or 🟡, save it to **💾 Saved Jobs**
    5. Go to **✂️ Tailor Resume** → select that job → generate a custom summary and bullet points
    6. Copy the tailored content into your actual resume/application
    7. Submit your application on the original site
    8. Come back and update the status in **💾 Saved Jobs** (New → Applied)
    9. Keep updating as you hear back (Applied → Interview → Offer / Rejected)

    ---

    ### What You Need

    | Requirement | Required? | Notes |
    |---|---|---|
    | **LLM API key** (OpenAI, xAI, or DeepSeek) | ✅ Yes | At least one — powers all AI features |
    | **Adzuna API keys** | ❌ Optional | Only for API Search tab. Free at [developer.adzuna.com](https://developer.adzuna.com) |

    **The app works perfectly without Adzuna** — the Paste Job tab is the
    primary workflow and only needs an LLM key.

    ---

    ### File Upload Support

    | Format | Status |
    |---|---|
    | **.txt** | ✅ Fully supported |
    | **.pdf** | ✅ Supported (text-based PDFs — scanned/image PDFs may not extract) |
    | **.docx** | ✅ Supported (Word 2007+) |
    | **.doc** | ⚠️ Not supported — re-save as .docx or .pdf first |
    | **.odt** | ⚠️ Not supported — re-save as .docx or .pdf first |

    ---

    ### Tips

    - **Better resume = better scores.** Include specific skills, tools, years
      of experience, and quantified achievements
    - **Paste the FULL description** — partial descriptions give less accurate scores
    - **Try different LLM models** — scores can vary between providers; compare if unsure
    - **Tailor before applying** — even small customization makes a real difference
    - **Score is a guide, not gospel** — use it to prioritize, but always read the
      job description yourself
    - **Track everything** — update statuses so you never lose track of applications

    ---

    ### Troubleshooting

    | Problem | Fix |
    |---|---|
    | "API key not found" | Add your key to `.streamlit/secrets.toml` |
    | Adzuna search not working | It's optional — use Paste Job tab instead, or get free keys at developer.adzuna.com |
    | PDF upload shows no text | PDF may be image-based (scanned) — copy-paste the text manually |
    | .doc file not supported | Re-save as .docx in Word (File → Save As → Word Document .docx) |
    | Scores seem off | Try a different model, or add more detail to your resume |
    | LinkedIn description looks incomplete | Click "Show more" / "See more" on LinkedIn before copying |
    """)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TABS — Paste Job is first (works without Adzuna)            ║
# ╚══════════════════════════════════════════════════════════════╝

tab_manual, tab_search, tab_results, tab_saved, tab_tailor = st.tabs([
    "📋 Paste Job",
    "🔍 API Search (Optional)",
    "📊 Search Results",
    "💾 Saved Jobs",
    "✂️ Tailor Resume",
])


# ──────────────────────────────────────────
# TAB 1: Manual Job Paste (Primary workflow)
# ──────────────────────────────────────────
with tab_manual:
    st.header("📋 Score a Job")
    st.caption(
        "Copy a job description from any site (LinkedIn, Indeed, Glassdoor, "
        "company career pages, etc.) and score it against your resume."
    )

    m_title = st.text_input("Job Title", key="manual_title")
    mc1, mc2 = st.columns(2)
    with mc1:
        m_company = st.text_input("Company", key="manual_company")
    with mc2:
        m_location = st.text_input("Location (optional)", key="manual_location")
    m_url = st.text_input("Job URL (optional)", key="manual_url")
    m_desc = st.text_area(
        "Full Job Description",
        height=250,
        key="manual_desc",
        placeholder="Paste the full job description here…",
    )

    if st.button("📊 Score This Job", type="primary", key="btn_manual_score"):
        if not m_title or not m_desc:
            st.warning("Title and description are required.")
        elif not st.session_state.resume:
            st.warning("Paste your resume in the sidebar first.")
        else:
            client = safe_get_client(provider)
            with st.spinner("Scoring…"):
                s, r = ai_score_job(
                    client, model, st.session_state.resume, m_desc
                )
            st.session_state.manual_score = {
                "score": s,
                "reason": r,
                "job": parse_manual_job(m_title, m_company, m_desc, m_url, m_location),
            }

    # Display manual score result (persists across reruns)
    if st.session_state.manual_score:
        ms = st.session_state.manual_score
        s = ms["score"]
        dot = "🟢" if s >= 70 else ("🟡" if s >= 40 else "🔴")

        st.divider()
        st.markdown(f"### {dot} Score: {s}/100")
        st.write(f"**Reasoning:** {ms['reason']}")

        job = ms["job"]
        if db_job_exists(job["title"], job["company"], job["url"]):
            st.info("✅ This job is already saved.")
        else:
            if st.button("💾 Save to My Jobs", key="btn_save_manual"):
                jid = db_save_job(job, ms["score"], ms["reason"])
                st.success(f"Saved! (ID: {jid})")
                st.session_state.manual_score = None
                st.rerun()


# ──────────────────────────────────────────
# TAB 2: Adzuna API Search (Optional)
# ──────────────────────────────────────────
with tab_search:
    st.header("🔍 Search Jobs via Adzuna")
    st.caption(
        "**Optional** — requires free API keys from "
        "[developer.adzuna.com](https://developer.adzuna.com) "
        "(250 requests/month). You can skip this and use the Paste Job tab instead."
    )

    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        search_query = st.text_input(
            "Keywords",
            placeholder="Python Developer, Data Engineer, etc.",
            key="search_query",
        )
    with col2:
        search_country = st.text_input(
            "Country code",
            value="us",
            placeholder="us, gb, ca, au, de…",
            key="search_country",
        )
    with col3:
        search_max = st.number_input(
            "Max results",
            min_value=5,
            max_value=50,
            value=10,
            key="search_max",
        )

    if st.button("🚀 Search & Score", type="primary", key="btn_search"):
        if not search_query:
            st.warning("Enter a search query.")
        elif not st.session_state.resume:
            st.warning("Paste your resume in the sidebar first.")
        else:
            with st.spinner("Searching Adzuna…"):
                jobs, err = search_adzuna(search_query, search_country, search_max)

            if err:
                st.error(err)
            elif not jobs:
                st.info("No results found. Try different keywords or country.")
            else:
                client = safe_get_client(provider)
                st.info(f"Scoring {len(jobs)} jobs with **{provider} / {model}**…")
                progress = st.progress(0, text="Scoring jobs…")

                for i, job in enumerate(jobs):
                    s, r = ai_score_job(
                        client, model, st.session_state.resume, job["description"]
                    )
                    job["score"] = s
                    job["reason"] = r
                    progress.progress(
                        (i + 1) / len(jobs),
                        text=f"Scored {i + 1}/{len(jobs)}: {job['title'][:40]}…",
                    )

                progress.empty()
                jobs.sort(key=lambda x: x["score"], reverse=True)
                st.session_state.search_results = jobs
                st.success(
                    f"✅ Scored {len(jobs)} jobs! Go to the **Search Results** tab."
                )


# ──────────────────────────────────────────
# TAB 3: Search Results
# ──────────────────────────────────────────
with tab_results:
    st.header("📊 Search Results")

    if not st.session_state.search_results:
        st.info(
            "No search results yet. Use the **🔍 API Search** tab to find jobs, "
            "or use **📋 Paste Job** to score individual listings."
        )
    else:
        results = st.session_state.search_results
        st.caption(f"{len(results)} jobs scored")

        # Quick summary bar
        high = len([j for j in results if j["score"] >= 70])
        med = len([j for j in results if 40 <= j["score"] < 70])
        low = len([j for j in results if j["score"] < 40])
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Good fit (70+)", high)
        c2.metric("🟡 Possible (40-69)", med)
        c3.metric("🔴 Low match (<40)", low)

        st.divider()

        for i, job in enumerate(results):
            sc = job["score"]
            dot = "🟢" if sc >= 70 else ("🟡" if sc >= 40 else "🔴")

            with st.expander(
                f"{dot} **{sc}/100** — {job['title']}  @  {job.get('company', '?')}",
                expanded=(i < 3),
            ):
                info_parts = []
                if job.get("location") and job["location"] != "N/A":
                    info_parts.append(f"📍 {job['location']}")
                if job.get("salary"):
                    info_parts.append(f"💰 {job['salary']}")
                info_parts.append(f"Source: {job.get('source', 'N/A')}")
                st.caption("  •  ".join(info_parts))

                if job.get("url") and job["url"] != "#":
                    st.markdown(f"🔗 [View original listing]({job['url']})")

                st.write(f"**Fit reasoning:** {job.get('reason', 'N/A')}")

                desc = job.get("description", "")
                st.write(desc[:500] + ("…" if len(desc) > 500 else ""))

                if db_job_exists(
                    job["title"], job.get("company", ""), job.get("url", "")
                ):
                    st.caption("✅ Already saved")
                else:
                    if st.button("💾 Save to My Jobs", key=f"save_result_{i}"):
                        jid = db_save_job(job, job["score"], job.get("reason", ""))
                        st.success(f"Saved! (ID: {jid})")
                        st.rerun()

        st.divider()
        if st.button("🗑️ Clear Search Results", key="btn_clear_results"):
            st.session_state.search_results = []
            st.rerun()


# ──────────────────────────────────────────
# TAB 4: Saved Jobs (Database)
# ──────────────────────────────────────────
with tab_saved:
    st.header("💾 Saved Jobs")

    all_jobs = db_get_all_jobs()

    if not all_jobs:
        st.info(
            "No saved jobs yet. Use **📋 Paste Job** to score and save jobs."
        )
    else:
        # Summary metrics
        statuses = [j["status"] for j in all_jobs]
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total", len(all_jobs))
        m2.metric("🆕 New", statuses.count("New"))
        m3.metric("📨 Applied", statuses.count("Applied"))
        m4.metric("🎤 Interview", statuses.count("Interviewing"))
        m5.metric("🎉 Offer", statuses.count("Offer"))
        m6.metric("❌ Rejected", statuses.count("Rejected"))

        st.divider()

        # Filter
        status_filter = st.multiselect(
            "Filter by status:",
            STATUS_OPTIONS,
            default=["New", "Applied", "Interviewing"],
            key="saved_filter",
        )

        filtered = (
            [j for j in all_jobs if j["status"] in status_filter]
            if status_filter
            else all_jobs
        )

        if not filtered:
            st.info("No jobs match the selected filters.")
        else:
            st.caption(f"Showing {len(filtered)} of {len(all_jobs)} jobs")

        for job in filtered:
            sc = job["score"]
            dot = "🟢" if sc >= 70 else ("🟡" if sc >= 40 else "🔴")
            status_icon = {
                "New": "🆕", "Applied": "📨", "Interviewing": "🎤",
                "Offer": "🎉", "Rejected": "❌", "Withdrawn": "🚫",
            }.get(job["status"], "❓")

            with st.expander(
                f"{dot} {sc}/100  {status_icon} {job['status']}  —  "
                f"**{job['title']}** @ {job['company']}"
            ):
                col_info, col_actions = st.columns([3, 1])

                with col_info:
                    st.caption(
                        f"Source: {job['source']}  •  "
                        f"Added: {job['created_at'][:10]}  •  "
                        f"Updated: {job['updated_at'][:10]}"
                    )
                    if job.get("url") and job["url"] not in ("#", ""):
                        st.markdown(f"🔗 [View original listing]({job['url']})")
                    if job.get("location"):
                        st.caption(f"📍 {job['location']}")
                    if job.get("salary"):
                        st.caption(f"💰 {job['salary']}")

                    st.write(
                        f"**Score reasoning:** {job.get('score_reason', 'N/A')}"
                    )

                    desc = job.get("description", "")
                    st.write(desc[:400] + ("…" if len(desc) > 400 else ""))

                    if job.get("tailored_summary"):
                        st.markdown("---")
                        st.markdown("**📝 Tailored Summary:**")
                        st.info(job["tailored_summary"])
                    if job.get("tailored_bullets"):
                        st.markdown("**📋 Tailored Bullets:**")
                        st.info(job["tailored_bullets"])

                    notes = st.text_area(
                        "Notes",
                        value=job.get("notes", ""),
                        key=f"notes_{job['id']}",
                        height=80,
                        placeholder="Your notes about this application…",
                    )
                    if notes != job.get("notes", ""):
                        db_update_notes(job["id"], notes)

                with col_actions:
                    current_idx = (
                        STATUS_OPTIONS.index(job["status"])
                        if job["status"] in STATUS_OPTIONS
                        else 0
                    )
                    new_status = st.selectbox(
                        "Status",
                        STATUS_OPTIONS,
                        index=current_idx,
                        key=f"status_{job['id']}",
                    )
                    if new_status != job["status"]:
                        db_update_status(job["id"], new_status)
                        st.rerun()

                    if st.button("🗑️ Delete", key=f"del_{job['id']}"):
                        db_delete_job(job["id"])
                        st.rerun()


# ──────────────────────────────────────────
# TAB 5: Tailor Resume
# ──────────────────────────────────────────
with tab_tailor:
    st.header("✂️ Tailor Resume Content")
    st.caption("Generate a tailored summary and bullet points for a saved job.")

    if not st.session_state.resume:
        st.warning("⚠️ Paste your resume in the sidebar first.")
        st.stop()

    all_jobs_tailor = db_get_all_jobs()
    if not all_jobs_tailor:
        st.info(
            "Save some jobs first (from **📋 Paste Job** or **📊 Search Results**), "
            "then come back here."
        )
        st.stop()

    # Job selector
    options = {}
    for j in all_jobs_tailor:
        label = f"[{j['score']}/100] {j['title']} @ {j['company']}"
        options[label] = j["id"]

    selected_label = st.selectbox(
        "Select a saved job to tailor for:",
        list(options.keys()),
        key="tailor_select",
    )
    job_id = options[selected_label]
    job = db_get_job(job_id)

    if not job:
        st.error("Job not found in database.")
        st.stop()

    # Show job info
    sc = job["score"]
    dot = "🟢" if sc >= 70 else ("🟡" if sc >= 40 else "🔴")
    st.markdown(
        f"{dot} **{job['title']}** at **{job['company']}** — Score: {sc}/100"
    )

    with st.expander("📄 View full job description"):
        st.write(job["description"])

    # Generate buttons
    gen_col1, gen_col2 = st.columns(2)

    with gen_col1:
        if st.button(
            "✍️ Generate Summary", type="primary", key="btn_gen_summary"
        ):
            client = safe_get_client(provider)
            with st.spinner("Generating tailored summary…"):
                result = ai_tailor_summary(
                    client, model, st.session_state.resume, job["description"]
                )
            st.session_state[f"gen_sum_{job_id}"] = result

    with gen_col2:
        if st.button(
            "📋 Generate Bullets", type="primary", key="btn_gen_bullets"
        ):
            client = safe_get_client(provider)
            with st.spinner("Generating tailored bullets…"):
                result = ai_tailor_bullets(
                    client, model, st.session_state.resume, job["description"]
                )
            st.session_state[f"gen_bul_{job_id}"] = result

    # Display generated or previously saved content
    sum_key = f"gen_sum_{job_id}"
    bul_key = f"gen_bul_{job_id}"

    sum_text = st.session_state.get(sum_key, job.get("tailored_summary", ""))
    bul_text = st.session_state.get(bul_key, job.get("tailored_bullets", ""))

    edited_sum = ""
    edited_bul = ""

    if sum_text:
        st.subheader("📝 Tailored Summary")
        edited_sum = st.text_area(
            "Edit summary if needed:",
            value=sum_text,
            height=150,
            key=f"edit_sum_{job_id}",
        )

    if bul_text:
        st.subheader("📋 Tailored Bullet Points")
        edited_bul = st.text_area(
            "Edit bullets if needed:",
            value=bul_text,
            height=200,
            key=f"edit_bul_{job_id}",
        )

    # Save tailored content
    if edited_sum or edited_bul:
        if st.button(
            "💾 Save Tailored Content to Database", key="btn_save_tailored"
        ):
            db_update_tailored(job_id, edited_sum, edited_bul)
            st.success("✅ Tailored content saved!")
            if sum_key in st.session_state:
                del st.session_state[sum_key]
            if bul_key in st.session_state:
                del st.session_state[bul_key]
            st.rerun()
    elif not sum_text and not bul_text:
        st.info(
            "Click **Generate Summary** or **Generate Bullets** above to "
            "create tailored resume content for this job."
        )
