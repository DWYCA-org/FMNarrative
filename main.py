import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set.")
client = Groq(api_key=api_key)

# Collecting data
home_team = input("Enter home team name: ")
away_team = input("Enter away team name: ")
home_or_away = input("Is this a home or away match for your team?: ")
score = input("Enter match score (e.g., 2-1): ")

home_goal_scorers = input("Enter home team goal scorers separated by commas: ").split(",")
away_goal_scorers = input("Enter away team goal scorers separated by commas: ").split(",")
home_xg = input(f"Enter expected goals (xG) for {home_team}: ")
away_xg = input(f"Enter expected goals (xG) for {away_team}: ")
pos_home = input(f"Enter possession % for {home_team}: ")
pos_away = input(f"Enter possession % for {away_team}: ")
shots_home = input(f"Enter shots on target for {home_team}: ")
shots_away = input(f"Enter shots on target for {away_team}: ")
cards_home = input(f"Enter number of cards for {home_team}: ")
cards_away = input(f"Enter number of cards for {away_team}: ")

# Build the prompt
prompt = f"""
You are a football manager press conference reporter.
Match: {home_team} {score} {away_team} ({home_or_away})
Home goal scorers: {', '.join([scorer.strip() for scorer in home_goal_scorers])}
Away goal scorers: {', '.join([scorer.strip() for scorer in away_goal_scorers])}
Home xG: {home_xg}
Away xG: {away_xg}
Possession: {pos_home}% vs {pos_away}%
Shots on target: {shots_home} vs {shots_away}
Cards: {cards_home} vs {cards_away}

Generate 3-5 realistic press conference questions a reporter might ask based on these stats.
"""

# Generate questions
response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="llama3-70b-8192",  # Replace with the appropriate model ID
)

print("Press Conference Questions:")
print(response.choices[0].message.content)
