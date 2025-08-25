import os
from groq import Groq
from dotenv import load_dotenv
import subprocess
from typing import Optional, Dict, Tuple

load_dotenv()

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if api_key is None:
    raise ValueError("GROQ_API_KEY environment variable not set.")
client = Groq(api_key=api_key)


def _try_run_ocr(image_path: str) -> Optional[Dict[str, object]]:
    """
    Run the C++ OCR binary (expected at ./build/ocr or ./ocr) and parse its output.
    Returns a dict with keys: home_team, away_team, stats (mapping stat -> (home, away)).
    On failure, returns None.
    """
    candidates = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "build/ocr")),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "ocr")),
    ]
    binary = next((p for p in candidates if os.path.exists(p) and os.access(p, os.X_OK)), None)
    if binary is None:
        return None

    try:
        proc = subprocess.run([binary, image_path], capture_output=True, text=True, check=False)
    except Exception:
        return None

    if proc.returncode != 0:
        return None

    home_team = None
    away_team = None
    stats: Dict[str, Tuple[str, str]] = {}

    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("HOME_TEAM:"):
            home_team = line.split(":", 1)[1].strip()
        elif line.startswith("AWAY_TEAM:"):
            away_team = line.split(":", 1)[1].strip()
        elif line.startswith("STAT:"):
            payload = line.split(":", 1)[1]
            parts = payload.split("|")
            if len(parts) >= 3:
                stat_name = parts[0].strip().lower()
                home_val = parts[1].strip()
                away_val = parts[2].strip()
                stats[stat_name] = (home_val, away_val)

    if not home_team and not away_team and not stats:
        return None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "stats": stats,
    }


def _parse_numeric(value: str) -> Optional[float]:
    """Parse numeric values from OCR output, handling percentages and complex formats."""
    if not value:
        return None
    try:
        cleaned = value.strip()
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1]
        # Handle forms like "12 (5/10)" -> "12"
        cleaned = cleaned.split(" ")[0]
        return float(cleaned)
    except Exception:
        return None


def extract_score_from_stats(stats: Dict[str, Tuple[str, str]]) -> Optional[Tuple[int, int]]:
    """Try to extract the match score from stats if available."""
    # Look for goals or score in the stats
    for key in ['goals', 'score', 'final score']:
        if key in stats:
            home_val, away_val = stats[key]
            try:
                return int(_parse_numeric(home_val) or 0), int(_parse_numeric(away_val) or 0)
            except:
                continue
    return None


def main():
    # Get screenshot - REQUIRED
    image_path = input("Enter the path to your match screenshot: ").strip()
    if not image_path:
        print("Screenshot path is required!")
        return
    
    # Run OCR
    print("Processing screenshot...")
    ocr_data = _try_run_ocr(image_path)
    
    if not ocr_data:
        print("Failed to extract data from screenshot. Please check the image path and OCR setup.")
        return
    
    # Extract basic data from OCR
    home_team = ocr_data.get("home_team", "Home Team")
    away_team = ocr_data.get("away_team", "Away Team")
    stats = ocr_data.get("stats", {})
    
    # Handle team name fallback
    if home_team in ["UNKNOWN", ""] or away_team in ["UNKNOWN", ""]:
        print("OCR couldn't detect team names properly.")
        home_team = input("Enter home team name: ").strip() or "Home Team"
        away_team = input("Enter away team name: ").strip() or "Away Team"
    
    print(f"\nDetected match: {home_team} vs {away_team}")
    print("Extracted stats:", list(stats.keys()))
    
    # Try to extract score from OCR
    score_tuple = extract_score_from_stats(stats)
    if score_tuple:
        home_score, away_score = score_tuple
        print(f"Detected score: {home_score}-{away_score}")
    else:
        # Fallback - ask for score if not found
        try:
            home_score = int(input(f"Enter {home_team} goals: "))
            away_score = int(input(f"Enter {away_team} goals: "))
        except ValueError:
            print("Invalid score input")
            return
    
    total_goals = home_score + away_score
    score = f"{home_score}-{away_score}"
    
    # MINIMAL USER INPUTS - Only what can't be extracted from screenshot
    
    # 1. Match setting and stage
    print("\n--- Match Context (cannot be determined from screenshot) ---")
    setting = input("Match setting (ucl/domestic cup/derby): ").lower().strip()
    
    if setting == "ucl":
        stage = input("UCL stage (league phase/ro32/ro16/qf/sf/final): ").lower().strip()
        stage_scores = {
            'league phase': 2, 'ro32': 4, 'ro16': 5, 
            'qf': 6, 'sf': 8, 'final': 10
        }
        importance = stage_scores.get(stage, 2)
    elif setting == "domestic cup":
        stage = input("Cup stage (qf/sf/final): ").lower().strip()
        stage_scores = {'qf': 2, 'sf': 3, 'final': 4}
        importance = stage_scores.get(stage, 2)
    elif setting == "derby":
        stage = "league"
        importance = 5  # Will be calculated later based on points
    else:
        print("Invalid setting")
        return
    
    # 2. Home/away designation
    home_or_away = input("Is this a home, away, or neutral match for your team?: ").lower().strip()
    
    # Adjust importance for away UCL matches
    if setting == 'ucl' and home_or_away == 'away' and stage not in ['league phase', 'final']:
        importance *= 1.3
    
    # 3. Goal scorers (names not clearly visible in screenshot)
    home_goal_scorers = []
    away_goal_scorers = []
    
    if home_score > 0:
        scorers_input = input(f"Enter {home_team} goal scorers (comma-separated): ").strip()
        home_goal_scorers = [s.strip() for s in scorers_input.split(",") if s.strip()]
    
    if away_score > 0:
        scorers_input = input(f"Enter {away_team} goal scorers (comma-separated): ").strip()
        away_goal_scorers = [s.strip() for s in scorers_input.split(",") if s.strip()]
    
    # 4. Derby-specific inputs (league context not in screenshot)
    if setting == "derby":
        home_team_points = int(input(f"{home_team} points before the match: "))
        away_team_points = int(input(f"{away_team} points before the match: "))
        match_number = int(input("Match number in league (1-34): "))
        
        # Calculate derby importance
        point_diff_before = abs(home_team_points - away_team_points)
        
        if match_number <= 8:
            season_multiplier = 0.8
        elif match_number <= 24:
            season_multiplier = 1.0
        else:
            season_multiplier = 1.3
        
        if point_diff_before <= 3:
            base_importance = 7
        elif point_diff_before <= 10:
            base_importance = 5
        else:
            base_importance = 3
        
        importance = base_importance * season_multiplier
    
    # EXTRACT ALL OTHER DATA FROM OCR
    print("\n--- Extracting stats from screenshot ---")
    
    # Extract stats with fallbacks - now checking all available OCR stats
    def get_stat(key_variations, default_home=0, default_away=0):
        for key in key_variations:
            if key in stats:
                home_val = _parse_numeric(stats[key][0])
                away_val = _parse_numeric(stats[key][1])
                if home_val is not None and away_val is not None:
                    return home_val, away_val
        return default_home, default_away
    
    # All stats from OCR
    home_shots, away_shots = get_stat(['shots'], 0, 0)
    home_shots_target, away_shots_target = get_stat(['on target'], 0, 0)
    home_xg, away_xg = get_stat(['xg'], 0.0, 0.0)
    home_shots_off, away_shots_off = get_stat(['off target'], 0, 0)
    home_clear_chances, away_clear_chances = get_stat(['clear cut chances'], 0, 0)
    home_long_shots, away_long_shots = get_stat(['long shots'], 0, 0)
    pos_home, pos_away = get_stat(['possession'], 50, 50)
    home_corners, away_corners = get_stat(['corners'], 0, 0)
    home_fouls, away_fouls = get_stat(['fouls'], 0, 0)
    home_offsides, away_offsides = get_stat(['offsides'], 0, 0)
    home_passes_comp, away_passes_comp = get_stat(['passes completed'], 50, 50)
    home_crosses_comp, away_crosses_comp = get_stat(['crosses completed'], 0, 0)
    home_tackles_won, away_tackles_won = get_stat(['tackles won'], 50, 50)
    home_headers_won, away_headers_won = get_stat(['headers won'], 50, 50)
    home_yellow, away_yellow = get_stat(['yellow cards'], 0, 0)
    home_red, away_red = get_stat(['red cards'], 0, 0)
    home_rating, away_rating = get_stat(['average rating'], 6.5, 6.5)
    home_prog_passes, away_prog_passes = get_stat(['progressive passes'], 0, 0)
    home_sprints, away_sprints = get_stat(['high intensity sprints'], 0, 0)
    
    # Convert to integers where appropriate
    home_shots, away_shots = int(home_shots), int(away_shots)
    home_shots_target, away_shots_target = int(home_shots_target), int(away_shots_target)
    home_shots_off, away_shots_off = int(home_shots_off), int(away_shots_off)
    home_clear_chances, away_clear_chances = int(home_clear_chances), int(away_clear_chances)
    home_long_shots, away_long_shots = int(home_long_shots), int(away_long_shots)
    pos_home, pos_away = int(pos_home), int(pos_away)
    home_corners, away_corners = int(home_corners), int(away_corners)
    home_fouls, away_fouls = int(home_fouls), int(away_fouls)
    home_offsides, away_offsides = int(home_offsides), int(away_offsides)
    home_yellow, away_yellow = int(home_yellow), int(away_yellow)
    home_red, away_red = int(home_red), int(away_red)
    home_prog_passes, away_prog_passes = int(home_prog_passes), int(away_prog_passes)
    home_sprints, away_sprints = int(home_sprints), int(away_sprints)
    
    # Display extracted stats
    print(f"Total Shots: {home_shots} vs {away_shots}")
    print(f"Shots on Target: {home_shots_target} vs {away_shots_target}")
    print(f"Expected Goals (xG): {home_xg} vs {away_xg}")
    print(f"Shots off Target: {home_shots_off} vs {away_shots_off}")
    print(f"Clear Cut Chances: {home_clear_chances} vs {away_clear_chances}")
    print(f"Long Shots: {home_long_shots} vs {away_long_shots}")
    print(f"Possession: {pos_home}% vs {pos_away}%")
    print(f"Corners: {home_corners} vs {away_corners}")
    print(f"Fouls: {home_fouls} vs {away_fouls}")
    print(f"Offsides: {home_offsides} vs {away_offsides}")
    print(f"Passes Completed: {home_passes_comp}% vs {away_passes_comp}%")
    print(f"Crosses Completed: {home_crosses_comp}% vs {away_crosses_comp}%")
    print(f"Tackles Won: {home_tackles_won}% vs {away_tackles_won}%")
    print(f"Headers Won: {home_headers_won}% vs {away_headers_won}%")
    print(f"Yellow Cards: {home_yellow} vs {away_yellow}")
    print(f"Red Cards: {home_red} vs {away_red}")
    print(f"Average Rating: {home_rating} vs {away_rating}")
    print(f"Progressive Passes: {home_prog_passes} vs {away_prog_passes}")
    print(f"High Intensity Sprints: {home_sprints} vs {away_sprints}")
    
    # Calculate derived metrics
    
    # Entertainment value
    if total_goals == 0:
        entertainment = 0
    elif total_goals >= 7:
        entertainment = 10
    elif total_goals >= 4:
        entertainment = 5
    elif total_goals >= 2:
        entertainment = 2
    else:
        entertainment = 1
    
    entertainment_labels = {0: "Boring", 1: "Alright", 2: "Decently fun", 5: "Great game", 10: "Incredible match"}
    match_entertainment = entertainment_labels[entertainment]
    
    # Importance level
    importance_labels = {
        (0, 1): 'Low importance',
        (1, 2): 'Moderate importance',
        (2, 4): 'High importance',
        (4, 7): 'Very high importance',
        (7, float('inf')): 'Extremely high importance'
    }
    importance_level = next(label for (low, high), label in importance_labels.items() if low <= importance < high)
    
    # Match aggression
    red_cards_total = home_red + away_red
    yellow_cards_total = home_yellow + away_yellow
    aggression_score = yellow_cards_total + (red_cards_total * 2)
    
    if aggression_score == 0:
        match_aggression = 0
    elif aggression_score >= 10:
        match_aggression = 3
    elif aggression_score >= 5:
        match_aggression = 2
    else:
        match_aggression = 1
    
    card_labels = {0: "Peaceful", 1: "Regular", 2: "Heated", 3: "Brawl"}
    match_heat = card_labels[match_aggression]
    
    # Team aggression analysis
    home_aggression_score = home_yellow + (home_red * 3)
    away_aggression_score = away_yellow + (away_red * 3)
    
    if home_aggression_score > away_aggression_score:
        aggr_team = home_team
        calmer_team = away_team
        aggression_difference = home_aggression_score - away_aggression_score
    elif away_aggression_score > home_aggression_score:
        aggr_team = away_team
        calmer_team = home_team
        aggression_difference = away_aggression_score - home_aggression_score
    else:
        aggr_team = "Both teams equally aggressive"
        calmer_team = "Both teams equally calm"
        aggression_difference = 0
    
    if aggression_difference == 0:
        aggression_analysis = "Both teams showed similar discipline"
    elif aggression_difference <= 2:
        aggression_analysis = f"{aggr_team} was slightly more aggressive than {calmer_team}"
    elif aggression_difference <= 5:
        aggression_analysis = f"{aggr_team} was noticeably more aggressive than {calmer_team}"
    else:
        aggression_analysis = f"{aggr_team} was significantly more aggressive than {calmer_team}"
    
    # Calculate shot efficiency
    home_shot_accuracy = (home_shots_target / home_shots * 100) if home_shots > 0 else 0
    away_shot_accuracy = (away_shots_target / away_shots * 100) if away_shots > 0 else 0
    
    # Calculate xG efficiency (goals vs expected goals)
    home_xg_efficiency = (home_score / home_xg * 100) if home_xg > 0 else 0
    away_xg_efficiency = (away_score / away_xg * 100) if away_xg > 0 else 0
    
    # Generate press conference questions
    print("\n--- Generating press conference questions ---")
    
    prompt = f"""
You are a football press conference reporter.

MATCH CONTEXT:
Setting: {setting.upper()}
Importance: {importance:.1f}/10 ({importance_level})
Match: {home_team} {score} {away_team} ({home_or_away})
The manager is managing the {home_or_away} team

COMPREHENSIVE MATCH STATS:
Goals: {home_score} vs {away_score}
Total Shots: {home_shots} vs {away_shots}
Shots on Target: {home_shots_target} vs {away_shots_target}
Shot Accuracy: {home_shot_accuracy:.1f}% vs {away_shot_accuracy:.1f}%
Expected Goals (xG): {home_xg} vs {away_xg}
xG Efficiency: {home_xg_efficiency:.1f}% vs {away_xg_efficiency:.1f}%
Clear Cut Chances: {home_clear_chances} vs {away_clear_chances}
Long Shots: {home_long_shots} vs {away_long_shots}
Possession: {pos_home}% vs {pos_away}%
Corners: {home_corners} vs {away_corners}
Fouls: {home_fouls} vs {away_fouls}
Offsides: {home_offsides} vs {away_offsides}
Pass Completion: {home_passes_comp}% vs {away_passes_comp}%
Cross Completion: {home_crosses_comp}% vs {away_crosses_comp}%
Tackles Won: {home_tackles_won}% vs {away_tackles_won}%
Headers Won: {home_headers_won}% vs {away_headers_won}%
Yellow Cards: {home_yellow} vs {away_yellow}
Red Cards: {home_red} vs {away_red}
Average Rating: {home_rating} vs {away_rating}
Progressive Passes: {home_prog_passes} vs {away_prog_passes}
High Intensity Sprints: {home_sprints} vs {away_sprints}

GOAL SCORERS:
Home goal scorers: {', '.join(home_goal_scorers) if home_goal_scorers else 'None'}
Away goal scorers: {', '.join(away_goal_scorers) if away_goal_scorers else 'None'}

MATCH CHARACTER:
Team Discipline: {aggression_analysis}
Match Atmosphere: {match_heat}
Entertainment Value: {match_entertainment}

Generate 5 realistic press conference questions a reporter might ask based on these comprehensive stats and the match importance.

GUIDELINES:
- You are an experienced, insightful football journalist who asks probing questions that reveal deeper truths
- Use the detailed statistics to ask specific, data-driven questions about tactical decisions and performance
- Focus on the psychological, tactical, and emotional aspects of the match rather than basic facts
- Ask about decision-making, team mentality, pressure handling, and strategic choices
- Consider the broader narrative: What does this result mean for the season? How does it change expectations?
- DO NOT invent or assume details not provided (like the fashion in which goals were scored or specific incidents)
- Base questions on the comprehensive stats provided, exploring their deeper tactical and psychological implications
- Ask questions that would make a manager think deeply about their approach and decisions
- Consider efficiency metrics (shot accuracy, xG efficiency) and what they reveal about performance
- Make each question feel like it comes from someone who truly understands football's tactical complexities

QUESTION THEMES TO EXPLORE:
- How tactical decisions influenced statistical outcomes (possession vs shots, defensive vs attacking approach)
- The correlation between stats and result (high possession but low shots, good xG but poor finishing, etc.)
- Physical and tactical intensity revealed by sprints, tackles, headers
- Team discipline and its impact on the match flow
- Efficiency in different phases of play (attacking, defending, set pieces)
- The psychological impact of the statistical performance on team confidence

Only ask the questions - no introduction or commentary.
"""
    
    # Generate questions
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-70b-8192"
    )
    
    print("\n=== PRESS CONFERENCE QUESTIONS ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()