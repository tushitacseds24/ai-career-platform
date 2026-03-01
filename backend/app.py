import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Load NLP model
nlp = spacy.load("en_core_web_sm")

app = FastAPI()

# -----------------------------
# ROLE SKILL REQUIREMENTS
# -----------------------------

ROLES = {
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React"],
    "ML Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow"]
}

# -----------------------------
# COURSE DATABASE
# -----------------------------

COURSES = {
    "Machine Learning": [
        {
            "name": "Machine Learning by Andrew Ng",
            "platform": "Coursera",
            "level": "Beginner",
            "duration_weeks": 8
        },
        {
            "name": "Google ML Crash Course",
            "platform": "Google",
            "level": "Beginner",
            "duration_weeks": 4
        }
    ],
    "Statistics": [
        {
            "name": "Statistics for Data Science",
            "platform": "Udemy",
            "level": "Beginner",
            "duration_weeks": 6
        }
    ],
    "Deep Learning": [
        {
            "name": "Deep Learning Specialization",
            "platform": "Coursera",
            "level": "Intermediate",
            "duration_weeks": 12
        }
    ]
}

# -----------------------------
# GLOBAL SKILL VOCABULARY
# -----------------------------

ALL_SKILLS = [
    "Python",
    "SQL",
    "Machine Learning",
    "Statistics",
    "HTML",
    "CSS",
    "JavaScript",
    "React",
    "Deep Learning",
    "TensorFlow"
]

# -----------------------------
# INPUT MODEL
# -----------------------------

class SkillRequest(BaseModel):
    user_skills: List[str] = []
    resume_text: str = ""
    target_role: str

# -----------------------------
# ROOT ROUTE (Health Check)
# -----------------------------

@app.get("/")
def root():
    return {"message": "AI Career Platform Backend Running"}

# -----------------------------
# ANALYZE ENDPOINT
# -----------------------------

@app.post("/analyze")
def analyze_skills(data: SkillRequest):

    # Check if role exists
    if data.target_role not in ROLES:
        return {"error": "Role not found"}

    # -----------------------------
    # NLP SKILL EXTRACTION
    # -----------------------------
    if data.resume_text:

        doc = nlp(data.resume_text)
        extracted_skills = []

        # Extract from noun phrases
        for chunk in doc.noun_chunks:
            for skill in ALL_SKILLS:
                if skill.lower() in chunk.text.lower():
                    extracted_skills.append(skill)

        # Fallback full-text matching
        for skill in ALL_SKILLS:
            if skill.lower() in data.resume_text.lower():
                extracted_skills.append(skill)

        # Remove duplicates
        extracted_skills = list(set(extracted_skills))

        # Merge with manual skills
        data.user_skills = list(set(data.user_skills + extracted_skills))

    # -----------------------------
    # SKILL GAP ANALYSIS
    # -----------------------------
    required_skills = ROLES[data.target_role]

    matched_skills = [
        skill for skill in required_skills
        if skill in data.user_skills
    ]

    missing_skills = [
        skill for skill in required_skills
        if skill not in data.user_skills
    ]

    # -----------------------------
    # READINESS SCORE
    # -----------------------------
    total_required = len(required_skills)
    matched_count = len(matched_skills)
    readiness_score = (matched_count / total_required) * 100

    # -----------------------------
    # COURSE RECOMMENDATION (OPTIMAL)
    # -----------------------------
    recommended_courses = {}

    for skill in missing_skills:
        if skill in COURSES:

            beginner_courses = [
                course for course in COURSES[skill]
                if course["level"] == "Beginner"
            ]

            if beginner_courses:
                sorted_courses = sorted(
                    beginner_courses,
                    key=lambda x: x["duration_weeks"]
                )
                recommended_courses[skill] = sorted_courses[0]
            else:
                recommended_courses[skill] = "No beginner course found"
        else:
            recommended_courses[skill] = "No course found"

    # -----------------------------
    # RETURN RESPONSE
    # -----------------------------
    return {
        "target_role": data.target_role,
        "extracted_user_skills": data.user_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "readiness_score": round(readiness_score, 2),
        "recommended_courses": recommended_courses
    }