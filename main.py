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

home_score=int(input("Enter home team goals: "))
away_score=int(input("Enter away goal goals: "))
total_goals=home_score+away_score
score=(f"{home_score}-{away_score}")
entertainment=0

if total_goals==0:
    entertainment=0
elif total_goals>=7:
    entertainment=10
elif total_goals>=4:
    entertainment=5
elif total_goals>=2:
    entertainment=2
else:
    entertainment=1

entertainment_labels={0:"Boring",1:"Alright",2:"Decently fun",5:"Great game",10:"Incredible match"}
match_entertainment=entertainment_labels[entertainment]

home_goal_scorers = input("Enter home team goal scorers separated by commas: ").split(",")
away_goal_scorers = input("Enter away team goal scorers separated by commas: ").split(",")
home_xg = float(input(f"Enter expected goals (xG) for {home_team}: "))
away_xg = float(input(f"Enter expected goals (xG) for {away_team}: "))
pos_home = int(input(f"Enter possession % for {home_team}: "))
pos_away = int(input(f"Enter possession % for {away_team}: "))
shots_home = int(input(f"Enter shots on target for {home_team}: "))
shots_away = int(input(f"Enter shots on target for {away_team}: "))

# Match aggression algorithm
red_cards_home=int(input(f"Enter number of red cards for {home_team}: "))
red_cards_away=int(input(f"Enter number of red cards for {away_team}: "))
red_cards_total=red_cards_home+red_cards_away
yellow_cards_home = int(input(f"Enter number of yellow cards for {home_team}: "))
yellow_cards_away = int(input(f"Enter number of yellow cards for {away_team}: "))
yellow_cards_total=yellow_cards_away+yellow_cards_home
cards_total=yellow_cards_away+yellow_cards_home+red_cards_away+red_cards_home
match_aggression=0
home_team_number=11-red_cards_home
away_team_number=11-red_cards_away

aggression_score=yellow_cards_total+(red_cards_total)*2
if aggression_score==0:
    match_aggression=0
elif aggression_score>=10:
    match_aggression=3
elif aggression_score>=5:
    match_aggression=2
elif match_aggression<5 and red_cards_total==0:
    match_aggression=1

card_labels={0: "Peaceful", 1: "Regular", 2: "Heated", 3: "Brawl"}
match_heat=card_labels[match_aggression]

home_aggression_score=yellow_cards_home+(red_cards_home)*3
away_aggression_score=yellow_cards_away+(red_cards_away)*3

if home_aggression_score>away_aggression_score:
    aggr_team=home_team
    calmer_team=away_team
    aggression_difference=home_aggression_score-away_aggression_score
elif away_aggression_score>home_aggression_score:
    aggr_team=away_team
    calmer_team=home_team
    aggression_difference=away_aggression_score-home_aggression_score
else:
    aggr_team="Both teams equally aggressive"
    calmer_team="Both teams equally calm"
    aggression_difference=0

if aggression_difference==0:
    aggression_analysis="Both teams showed similar discipline"
elif aggression_difference<=2:
    aggression_analysis=f"{aggr_team} was slightly more aggressive than {calmer_team}"
elif aggression_difference<=5:
    aggression_analysis=f"{aggr_team} was noticeably more aggressive than {calmer_team}"
else:
    aggression_analysis=f"{aggr_team} was significantly more aggressive than {calmer_team}"


# Build the prompt
prompt = f"""
You are a football press conference reporter.
Match: {home_team} {home_score} {score} {away_team} {away_score} ({home_or_away}) 
Home goal scorers: {', '.join([scorer.strip() for scorer in home_goal_scorers])}
Away goal scorers: {', '.join([scorer.strip() for scorer in away_goal_scorers])}
Home xG: {home_xg}
Away xG: {away_xg}
Possession: {pos_home}% vs {pos_away}%
Shots on target: {shots_home} vs {shots_away}
Yellow cards: {yellow_cards_home} vs {yellow_cards_away}
Red cards: {red_cards_home} vs {red_cards_away}
Team Discipline: {aggression_analysis}
Most aggressive team: {aggr_team}
Calmest team: {calmer_team}
{match_heat} denotes how aggressive the game went
{match_entertainment} denotes how fun of a viewer experience the match provided.


Generate 3-5 realistic press conference questions a reporter might ask based on these stats.
"""

# Generate questions
response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="llama3-70b-8192",  # Replace with the appropriate model ID
)

print("Press Conference Questions:")
print(response.choices[0].message.content)
