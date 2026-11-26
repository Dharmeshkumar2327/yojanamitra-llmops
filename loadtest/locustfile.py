"""Load test.

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
           --users 20 --spawn-rate 2 --run-time 3m --headless --csv loadtest/report

Mix of repeated questions (exercise the semantic cache) and unique ones.
NOTE: raise RATE_LIMIT_PER_MIN on the server while load testing, and keep runs short
to stay inside the free Groq quota.
"""

import random
import uuid

from locust import HttpUser, between, task

QUESTIONS = [
    "Who is eligible for PM-KISAN?",
    "How much money does a farmer get under PM-KISAN?",
    "What is the premium of PMSBY?",
    "What is the age limit for Atal Pension Yojana?",
    "How much loan can I get under Mudra Shishu?",
    "Who can open a Sukanya Samriddhi account?",
    "आयुष्मान भारत योजना में कितने रुपये तक का इलाज मुफ्त है?",
    "What documents are required for PM Ujjwala Yojana?",
]


class Citizen(HttpUser):
    wait_time = between(1, 4)

    def on_start(self):
        self.session_id = uuid.uuid4().hex[:16]

    @task(5)
    def ask_common(self):
        self.client.post(
            "/ask",
            json={"question": random.choice(QUESTIONS), "session_id": self.session_id},
            name="/ask (common)",
        )

    @task(1)
    def ask_unique(self):
        q = f"{random.choice(QUESTIONS)} I live in district {random.randint(1, 700)}."
        self.client.post(
            "/ask", json={"question": q, "session_id": self.session_id}, name="/ask (unique)"
        )

    @task(1)
    def health(self):
        self.client.get("/health")
