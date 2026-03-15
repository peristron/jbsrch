import streamlit as st
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
import re

# ---------- Password Protection ----------
def check_password():
    """Returns `True` if the user entered the correct password."""
    def password_entered():
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show password input
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Wrong password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()  # Do not continue if not authenticated

# ---------- LLM Client Factory ----------
def get_llm_client(llm_choice):
    """Return an OpenAI-compatible client for the chosen LLM."""
    try:
        if llm_choice == "OpenAI":
            return OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        elif llm_choice == "xAI Grok":
            return OpenAI(
                api_key=st.secrets["XAI_API_KEY"],
                base_url="https://api.x.ai/v1/"
            )
        elif llm_choice == "DeepSeek":
            return OpenAI(
                api_key=st.secrets["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com/v1/"
            )
        else:
            st.error("Invalid LLM choice")
            st.stop()
    except KeyError as e:
        st.error(f"Missing API key for {llm_choice}. Please check your secrets.")
        st.stop()

# ---------- Job Scraper (Demo only – replace with official APIs) ----------
def scrape_indeed_jobs(query, location="USA", max_jobs=5):
    """
    Basic scraper for Indeed. May break or be blocked.
    For production, use official job APIs (Adzuna, etc.) or services like SerpAPI.
    """
    base_url = "https://www.indeed.com/jobs"
    params = {"q": query, "l": location}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
    except Exception:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    job_cards = soup.find_all("div", class_="job_seen_beacon")[:max_jobs]
    jobs = []

    for card in job_cards:
        title_elem = card.find("h2", class_="jobTitle")
        title = title_elem.text.strip() if title_elem else "N/A"
        company_elem = card.find("span", class_="companyName")
        company = company_elem.text.strip() if company_elem else "N/A"
        desc_elem = card.find("div", class_="job-snippet")
        description = desc_elem.text.strip() if desc_elem else "N/A"
        link_elem = card.find("a", class_="jcs-JobTitle")
        url = "https://www.indeed.com" + link_elem["href"] if link_elem else "N/A"

        jobs.append({
            "title": title,
            "company": company,
            "description": description,
            "url": url
        })
    return jobs

# ---------- LLM Scoring ----------
def score_job_fit(client, model, user_profile, job_description):
    prompt = f"""
    User profile: {user_profile}
    
    Job description: {job_description}
    
    Score how well the user's profile fits this job on a scale of 0-100, where 100 is perfect match.
    Provide only the score as an integer, nothing else.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0.2
        )
        score_text = response.choices[0].message.content.strip()
        match = re.search(r'\d+', score_text)
        return int(match.group()) if match else 0
    except Exception as e:
        st.warning(f"Scoring failed: {e}")
        return 0

# ---------- Resume Summary Tailoring ----------
def tailor_resume_summary(client, model, user_profile, job_description):
    prompt = f"""
    User profile: {user_profile}
    
    Job description: {job_description}
    
    Generate a concise, keyword-optimized professional summary for the user's resume tailored to this job.
    Keep it to 4-6 sentences.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.warning(f"Tailoring failed: {e}")
        return ""

# ---------- Streamlit UI ----------
st.set_page_config(page_title="JobOps Python POC", page_icon="💼")
st.title("💼 JobOps Python POC: AI Job Scorer & Resume Tailorer")
st.markdown("Prototype combining job scraping, AI scoring, and resume tailoring with your choice of LLM.")

# LLM selection
llm_choice = st.selectbox("Select LLM:", ["OpenAI", "xAI Grok", "DeepSeek"])

model_map = {
    "OpenAI": "gpt-3.5-turbo",
    "xAI Grok": "grok-beta",
    "DeepSeek": "deepseek-chat"
}
model = model_map[llm_choice]

# Get client (will stop if key missing)
client = get_llm_client(llm_choice)

# User inputs
user_profile = st.text_area("📄 Enter your resume/profile text (skills, experience, etc.):", height=200)
job_query = st.text_input("🔍 Job search query (e.g., 'Python Developer'):")
location = st.text_input("📍 Location (e.g., 'Remote' or 'New York'):", value="USA")

if st.button("🚀 Search & Score Jobs"):
    if not user_profile or not job_query:
        st.error("Please provide your profile and job query.")
    else:
        with st.spinner("Scraping jobs (this may fail if Indeed blocks us)..."):
            jobs = scrape_indeed_jobs(job_query, location)

        if not jobs:
            st.warning("No jobs found or scraping failed. For a reliable demo, consider using a job API (e.g., Adzuna) instead of scraping.")
            # Optional: add fallback mock jobs for demonstration
            if st.checkbox("Use mock job data for demo"):
                jobs = [
                    {"title": "Python Developer", "company": "Tech Corp", "description": "Looking for a Python expert with Django experience.", "url": "#"},
                    {"title": "Data Scientist", "company": "Data Inc", "description": "Need strong skills in ML and Python.", "url": "#"},
                ]
        else:
            st.success(f"Found {len(jobs)} jobs. Scoring with {llm_choice}...")

        if jobs:
            scored_jobs = []
            for job in jobs:
                score = score_job_fit(client, model, user_profile, job["description"])
                job["score"] = score
                scored_jobs.append(job)

            scored_jobs.sort(key=lambda x: x["score"], reverse=True)

            st.subheader("📊 Scored Job Listings")
            for job in scored_jobs:
                with st.expander(f"**{job['title']}** at {job['company']} (Score: {job['score']}/100)"):
                    st.write(job["description"])
                    if job["url"] != "N/A" and job["url"] != "#":
                        st.markdown(f"[View Job]({job['url']})")
                    else:
                        st.write("*(No URL available)*")
                    
                    if st.button(f"✨ Generate Tailored Summary", key=job['title']+job['company']):
                        summary = tailor_resume_summary(client, model, user_profile, job["description"])
                        st.subheader("Tailored Resume Summary")
                        st.write(summary if summary else "Generation failed.")
