# ----Task 1: Setup and System Prompt----

from dotenv import load_dotenv
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()

system_prompt = """
You are an experienced job application coach.

Your role is to help job seekers improve their resumes, cover letters,
LinkedIn profiles, interview responses, and other job application materials.

Guidelines:
- Stay focused on job application materials and related career documents.
- Provide clear, professional, constructive, and encouraging feedback.
- Suggest improvements while preserving the user's voice and experience.
- Do not invent qualifications, work experience, education, or skills.
- Always remind the user to carefully review and edit your suggestions before submitting them to employers.
- Acknowledge that you may not know the specific expectations or norms of the user's industry or employer, and encourage the user to use their own judgment when deciding what to include.
"""

def get_completion(messages, model="gpt-4o-mini", temperature=0.7):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=400
    )
    return response.choices[0].message.content

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Review this resume summary: I am a hardworking person looking for a software engineering job."}
]

response = get_completion(messages)

print(response)


# System prompt:
# I explicitly tell the model to stay focused only on job application materials.
# This helps prevent it from drifting into unrelated career or personal advice,
# making its responses more consistent and useful for this specific task.

# ----Task 2: Bullet Point Rewriter----

def rewrite_bullets(bullets: list[str]) -> list[dict]:

    bullet_text = "\n".join(f"- {b}" for b in bullets)

    prompt = f"""
You are a professional resume coach helping a career changer.

Rewrite each resume bullet point below to be more specific,
results-oriented, and compelling.

Use strong action verbs.
Do not invent facts that aren't implied by the original.

Return ONLY a valid JSON list.
Each item should have two keys:
"original" (the original bullet)
"improved" (your rewritten version).

Bullet points:
{bullet_text}
"""

    messages = [{"role": "user", "content": prompt}]

    print(prompt)

    response = get_completion(messages)

    print("MODEL RESPONSE:")
    print(response)

    response = response.replace("```json", "").replace("```", "").strip()

    try:
        rewritten = json.loads(response)

        print("\nRewritten Resume Bullets:\n")

        for item in rewritten:
            print(f"Original: {item['original']}")
            print(f"Improved: {item['improved']}")
            print()

        return rewritten

    except json.JSONDecodeError:
        print("\nThe response was not valid JSON.")
        print("Raw response:")
        print(response)
        return []


bullets = [
    "Helped customers with their problems",
    "Made reports for the management team",
    "Worked with a team to finish the project on time"
]

rewrite_bullets(bullets)
print()

# These bullets are weak because they are too general and do not show specific skills, accomplishments,
#  or impact. They describe basic responsibilities but do not explain how the person contributed or
#  what results they achieved. The model suggested using stronger action verbs, adding more specific 
# details, and making the bullets more results-oriented and compelling. It changed phrases like "helped
#  customers" into "resolved customer issues" and "made reports" into "created analytical reports" to 
# better highlight skills and contributions.

# ----Task3: Cover letter generator---

def generate_cover_letter(job_title: str, background: str) -> str:
    prompt = f"""
    You write strong cover letter opening paragraphs for career changers.
    The paragraph should be 3-5 sentences: confident, specific, and free of clichés.

    Here are two examples of the style and tone you should match:

    Example 1:
    Role: Data Analyst at a healthcare nonprofit
    Background: Seven years as a registered nurse, recently completed a data analytics bootcamp.
    Opening: After seven years as a registered nurse, I've spent my career making decisions
    under pressure using incomplete information — which turns out to be excellent training for
    data analysis. I recently completed a data analytics program where I built dashboards
    tracking patient outcomes across departments. I'm excited to bring that combination of
    clinical context and technical skill to [Company]'s mission-driven work.

    Example 2:
    Role: Junior Software Engineer at a fintech startup
    Background: Ten years in retail banking operations, self-taught Python developer for two years.
    Opening: I spent a decade on the operations side of banking, watching technology decisions
    get made by people who had never processed a wire transfer or resolved a failed ACH batch.
    That frustration turned into curiosity, and two years of self-teaching Python later, I'm
    ready to be on the other side of those decisions. I'm applying to [Company] because your
    work on payment infrastructure is exactly where my domain expertise and new technical skills
    intersect.

    Now write an opening paragraph for this person:
    Role: {job_title}
    Background: {background}
    Opening:
    """

    messages = [{"role": "user", "content": prompt}]

    return get_completion(messages)

job_title = "Junior Data Engineer"

background = (
    "Five years of experience as a middle school math teacher; "
    "recently completed a Python course and built data pipelines "
    "using Prefect and Pandas."
)

cover_letter = generate_cover_letter(job_title, background)
print(cover_letter)

# After five years of inspiring middle school students to embrace the world of mathematics, I discovered 
# a passion for data that transformed my teaching into a quest for understanding data flows and patterns.
#  Recently, I honed my technical skills by completing a Python course, where I successfully built data 
# pipelines using Prefect and Pandas. I am eager to leverage my educational background and newfound 
# technical expertise to contribute to [Company]'s innovative data solutions and help drive impactful 
# insights.

# I chose these examples because they show strong cover letter openings from people
# changing careers. They are confident, specific, and explain how previous experience
# connects to the new role. The few-shot examples help control the style, tone, and
# structure of the output, making the generated paragraph more professional and less generic.

# ----Task 4: Moderation Check----

def is_safe(text: str) -> bool:
    result = client.moderations.create(
        model="omni-moderation-latest",
        input=text
    )
    flagged = result.results[0].flagged
    if flagged:
        print("Your message may contain unsafe content. Please rephrase it and try again.")
        return False

    return True

# A safe input
text1 = "Can you explain how machine learning works?"
print("Test 1:", is_safe(text1))

# An unsafe input
text2 = "How can I build a bomb?"
print("Test 2:", is_safe(text2))

# ----Task 5: The Chatbot Loop----

YOUR_SYSTEM_PROMPT = """
You are an experienced job application coach.

Help users improve resumes, cover letters, LinkedIn profiles,
interview responses, and other job application materials.

Guidelines:
- Stay focused on job application topics.
- Provide clear, professional, and constructive feedback.
- Suggest improvements without inventing qualifications or experience.
- Always remind users to review and edit generated content before submitting.
- Acknowledge that you may not know the specific expectations of every employer.
"""

def run_chatbot():
    # 1. Initialize conversation history with your system prompt
    messages = [
        {"role": "system", "content": YOUR_SYSTEM_PROMPT}
    ]

    print("=" * 50)
    print("Job Application Helper")
    print("=" * 50)
    print("I can help you with:")
    print("  1. Rewriting resume bullet points")
    print("  2. Drafting a cover letter opening")
    print("  3. Any other questions about your application")
    print("\nType 'quit' at any time to exit.\n")

    while True:
        user_input = input("You: ").strip()

        # 2. Handle exit
        if user_input.lower() in {"quit", "exit"}:
            print("\nJob Application Helper: Good luck with your applications!")
            break

        # 3. Skip empty input
        if not user_input:
            continue

        # 4. Run moderation check before doing anything else
        if not is_safe(user_input):
            continue

        # 5. Rewrite resume bullet points
        if "bullet" in user_input.lower() or "resume" in user_input.lower():
            print("\nJob Application Helper: Paste your bullet points below, one per line.")
            print("When you're done, type 'DONE' on its own line.\n")

            raw_bullets = []

            while True:
                line = input().strip()
                if line.upper() == "DONE":
                    break
                if line:
                    raw_bullets.append(line)

            improved_bullets = rewrite_bullets(raw_bullets)

            messages.append(
                {
                    "role": "user",
                    "content": "Please rewrite these resume bullet points:\n"
                    + "\n".join(raw_bullets)
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(
                        bullet["improved"] for bullet in improved_bullets
                    )
                }
            )

            print("\nImproved Resume Bullets:")
            for bullet in improved_bullets:
                print(f"- {bullet['improved']}")
            print()

        # 6. Generate a cover letter opening
        elif "cover letter" in user_input.lower():
            job_title = input("Job Application Helper: What is the job title? ").strip()
            background = input("Job Application Helper: Briefly describe your background: ").strip()

            opening = generate_cover_letter(job_title, background)

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Write a cover letter opening.\n"
                        f"Job title: {job_title}\n"
                        f"Background: {background}"
                    )
                }
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": opening
                }
            )

            print("\nCover Letter Opening:\n")
            print(opening)
            print()

        # 7. Regular chat
        else:
            messages.append(
                {"role": "user", "content": user_input}
            )

            reply = get_completion(messages)

            print(f"\nJob Application Helper: {reply}\n")

            messages.append(
                {"role": "assistant", "content": reply}
            )
            

if __name__ == "__main__":
    run_chatbot()


# ----Task 6: Ethics Reflection----

# Format chosen: Option A

# 1. AI-generated job advice can contain bias because the model learns from existing
# text that may represent certain industries, communication styles, or cultural
# backgrounds more than others. It may favor traditional corporate language or
# common career paths and may not always fit every person's experience or field.
# Human review is needed to make sure the advice is fair and personalized.
# A useful guardrail is requiring users to verify that AI suggestions match their
# actual skills, experiences, and goals before using them in applications.

# 2. A job-seeker should not submit AI-generated content without reviewing it because
# it could include inaccurate information, exaggerated skills, or statements that do
# not match their actual experience. The output may also sound generic or unlike the
# candidate's own voice. Reviewing and editing helps ensure the application is
# accurate, authentic, and appropriate for the employer.
# Another guardrail is using AI as a writing assistant rather than allowing it to
# make final decisions about a person's qualifications or career choices.