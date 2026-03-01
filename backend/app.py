from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

ROLES = {
    "Data Scientist": ["Python", "SQL", "Machine Learning", "Statistics"],
    "Web Developer": ["HTML", "CSS", "JavaScript", "React"],
    "ML Engineer": ["Python", "Machine Learning", "Deep Learning", "TensorFlow"]
}

class SkillRequest(BaseModel):
    user_skills: List[str]
    target_role: str

@app.post("/analyze")
def analyze_skills(data: SkillRequest):

    if data.target_role not in ROLES:
        return {"error": "Role not found"}

    required_skills = ROLES[data.target_role]

    matched_skills = [
        skill for skill in required_skills
        if skill in data.user_skills
    ]

    missing_skills = [
        skill for skill in required_skills
        if skill not in data.user_skills
    ]

    total_required = len(required_skills)
    matched_count = len(matched_skills)

    readiness_score = (matched_count / total_required) * 100

    return {
        "target_role": data.target_role,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "readiness_score": round(readiness_score, 2)
    }