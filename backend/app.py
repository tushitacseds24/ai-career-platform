import spacy
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer, util

# Load NLP model
nlp = spacy.load("en_core_web_sm")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

app = FastAPI()

# -----------------------------
# ROLE SKILL REQUIREMENTS
# -----------------------------

ROLES = {
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React"],
    "ML Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow"]
}
ROLE_DESCRIPTIONS = {
    "Data Scientist": "Work with data, machine learning models, statistics, SQL databases, and Python programming.",
    "Web Developer": "Build websites using HTML, CSS, JavaScript and modern frontend frameworks like React.",
    "ML Engineer": "Develop machine learning systems using Python, deep learning, TensorFlow and scalable AI pipelines."
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

@app.post("/analyze-job-seeker")
def analyze_skills(data: SkillRequest):

    # -----------------------------
    # CHECK ROLE EXISTS
    # -----------------------------
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
    # RULE-BASED READINESS SCORE
    # -----------------------------
    total_required = len(required_skills)
    matched_count = len(matched_skills)
    readiness_score = (matched_count / total_required) * 100

    # -----------------------------
    # AI SEMANTIC MATCH SCORE
    # -----------------------------
    ai_match_score = 0

    if data.resume_text and data.target_role in ROLE_DESCRIPTIONS:

        resume_embedding = embedding_model.encode(
            data.resume_text,
            convert_to_tensor=True
        )

        role_embedding = embedding_model.encode(
            ROLE_DESCRIPTIONS[data.target_role],
            convert_to_tensor=True
        )

        similarity = util.pytorch_cos_sim(
            resume_embedding,
            role_embedding
        )

        ai_match_score = float(similarity[0][0]) * 100

    # -----------------------------
    # HYBRID FINAL SCORE
    # -----------------------------
    final_score = (readiness_score * 0.6) + (ai_match_score * 0.4)

    # -----------------------------
    # COURSE RECOMMENDATION
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
    # RESPONSE
    # -----------------------------
    return {
        "target_role": data.target_role,
        "extracted_user_skills": data.user_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "rule_based_readiness_score": round(readiness_score, 2),
        "ai_semantic_match_score": round(ai_match_score, 2),
        "final_ai_readiness_score": round(final_score, 2),
        "recommended_courses": recommended_courses
    }

@app.post("/analyze-upskiller")
def analyze_upskiller(data: SkillRequest):

    # Reuse the same logic for now
    result = analyze_skills(data)

    # Slight adjustment: boost AI weight for professionals
    if "final_ai_readiness_score" in result:
        boosted_score = result["final_ai_readiness_score"] * 1.05
        result["final_ai_readiness_score"] = round(min(boosted_score, 100), 2)

    result["mode"] = "Upskiller"

    return result

@app.post("/explore")
def explore_career(data: SkillRequest):

    suggestions = []

    for role, skills in ROLES.items():
        match_count = len([skill for skill in skills if skill in data.user_skills])
        suggestions.append({
            "role": role,
            "match_score": round((match_count / len(skills)) * 100, 2)
        })

    suggestions = sorted(suggestions, key=lambda x: x["match_score"], reverse=True)

    return {
        "mode": "Explorer",
        "career_suggestions": suggestions
    }