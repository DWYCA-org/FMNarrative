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
#Match importance check
setting=input("What was the match setting? (UCL, Domestic Cup, Derby): ").lower().strip()
importance=0

if setting=="ucl":
    home_team = input("Enter home team name: ")
    away_team = input("Enter away team name: ")
    home_or_away = input("Is this a home, away, or neutral match for your team?: ")
    stage=input("Enter the stage (League Phase, RO32, RO16, QF, SF, Final): ").lower().strip()
    
    stage_scores={
        'league phase': 2,
        'ro32':4,
        'ro16':5,
        'qf':6,
        'sf':8,
        'final':10
    }
    importance=stage_scores.get(stage,2)

    if home_or_away=='away' and stage not in ['league phase','final']:
        importance*=1.3

elif setting=="domestic cup":
    home_team = input("Enter home team name: ")
    away_team = input("Enter away team name: ")
    home_or_away = input("Is this a home, away, or neutral match for your team?: ")
    stage=input("Enter the stage (QF, SF, Final): ").lower().strip()
    
    stage_scores={
        'qf':2,
        'sf':3,
        'final':4
    }
    importance=stage_scores.get(stage,2)

elif setting=="derby":
    home_team = input("Enter home team name: ")
    away_team = input("Enter away team name: ")
    home_or_away = input("Is this a home or away match for your team?: ")
    home_team_points=int(input(f"{home_team} points before the match: "))
    away_team_points=int(input(f"{away_team} points before the match: "))
    title_favourite=max(home_team_points,away_team_points)
    point_diff_before=abs(home_team_points-away_team_points)

    match_number=int(input("What match number is this in the league? (1-34 total matches): "))

    if match_number<=8:
        season_stage='early season'
        season_multiplier=0.8
    elif match_number <=24:
        season_stage='mid-season'
        season_multiplier=1.0
    else:
        season_stage='late season'
        season_multiplier=1.3

    if point_diff_before<=3:
        base_importance=7
    elif point_diff_before<=10:
        base_importance=5
    else:
        base_importance=3

    importance=base_importance*season_multiplier

importance_labels={
    (0,1):'Low importance',
    (1,2):'Moderate importance',
    (2,4):'High importance',
    (4,7):'Very high importance',
    (7,float('inf')):'Extremely high importance'
}

importance_level=next(label for (low,high), label in importance_labels.items() if low <= importance <high)

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
elif aggression_score<5 and red_cards_total==0:
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

MATCH CONTEXT:
Setting: {setting.upper()}
Importance: {importance}/10 ({importance_level})
Match: {home_team} {score} {away_team} ({home_or_away})
The manager is managing {home_or_away}

MATCH STATS:
Home goal scorers: {', '.join([scorer.strip() for scorer in home_goal_scorers if scorer.strip()])}
Away goal scorers: {', '.join([scorer.strip() for scorer in away_goal_scorers if scorer.strip()])}
Expected Goals: {home_xg} vs {away_xg}
Possession: {pos_home}% vs {pos_away}%
Shots on target: {shots_home} vs {shots_away}
Yellow cards: {yellow_cards_home} vs {yellow_cards_away}
Red cards: {red_cards_home} vs {red_cards_away}

MATCH CHARACTER:
Team Discipline: {aggression_analysis}
Match Atmosphere: {match_heat}
Entertainment Value: {match_entertainment}

Generate 5 realistic press conference questions a reporter might ask based on these stats and the match importance.

GUIDELINES:
- You are an experienced, insightful football journalist who asks probing questions that reveal deeper truths
- Focus on the psychological, tactical, and emotional aspects of the match rather than basic facts
- Ask about decision-making, team mentality, pressure handling, and strategic choices
- Consider the broader narrative: What does this result mean for the season? How does it change expectations?
- DO NOT invent or assume details not provided (like the fashion in which goals were scored or specific incidents)
- Base questions on the stats and context given, but explore their deeper implications
- Ask questions that would make a manager think deeply about their approach and decisions
- Consider the human drama: leadership under pressure, team psychology, managing expectations
- Make each question feel like it comes from someone who truly understands football's complexities

QUESTION THEMES TO EXPLORE:
- How tactical decisions influenced the outcome
- The stage of the season in which this match was played (early season, mid-season, season's end)
- The psychological impact of the result on players and season trajectory  
- Leadership and decision-making under pressure
- Team mentality and character revealed by the performance
- Strategic implications for upcoming matches
- How this result fits into the bigger picture of the season

Only ask the questions - no introduction or commentary.
"""

# Generate questions
response = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="llama3-70b-8192",  # Replace with the appropriate model ID

)

print("Press Conference Questions:")
print(response.choices[0].message.content)
