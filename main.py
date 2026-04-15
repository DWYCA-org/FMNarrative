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

# Percentage fields accept values like "65" or "65%"; parsing strips the % symbol.
MANUAL_STAT_FIELDS = [
    ("shots", "total shots"),
    ("shots_on_target", "shots on target"),
    ("xg", "expected goals (xG)"),
    ("shots_off_target", "shots off target"),
    ("clear_cut_chances", "clear cut chances"),
    ("long_shots", "long shots"),
    ("possession", "possession (%)"),
    ("corners", "corners"),
    ("fouls", "fouls"),
    ("offsides", "offsides"),
    ("passes_completed", "passes completed (%)"),
    ("crosses_completed", "crosses completed (%)"),
    ("tackles_won", "tackles won (%)"),
    ("headers_won", "headers won (%)"),
    ("yellow_cards", "yellow cards"),
    ("red_cards", "red cards"),
    ("average_rating", "average rating"),
    ("progressive_passes", "progressive passes"),
    ("high_intensity_sprints", "high intensity sprints"),
]

STAT_KEY_ALIASES = {
    "shots on target": "shots_on_target",
    "on target": "shots_on_target",
    "shots off target": "shots_off_target",
    "off target": "shots_off_target",
    "clear cut chances": "clear_cut_chances",
    "clear-cut chances": "clear_cut_chances",
    "long shots": "long_shots",
    "passes completed": "passes_completed",
    "crosses completed": "crosses_completed",
    "tackles won": "tackles_won",
    "headers won": "headers_won",
    "yellow cards": "yellow_cards",
    "red cards": "red_cards",
    "average rating": "average_rating",
    "progressive passes": "progressive_passes",
    "high intensity sprints": "high_intensity_sprints",
}

CANCEL_INPUTS = {"q", "quit"}


class InputCancelled(Exception):
    pass


def _check_cancel(raw_value: str) -> None:
    if raw_value.strip().lower() in CANCEL_INPUTS:
        raise InputCancelled()


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
                stat_name = _normalize_stat_key(parts[0])
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


def _normalize_stat_key(raw_key: str) -> str:
    cleaned = raw_key.strip().lower()
    if cleaned in STAT_KEY_ALIASES:
        return STAT_KEY_ALIASES[cleaned]
    return cleaned.replace(" ", "_")


def _prompt_choice(prompt: str, choices: Dict[str, str]) -> str:
    while True:
        selection = input(prompt).strip()
        _check_cancel(selection)
        selection = selection.lower()
        if selection in choices:
            return choices[selection]
        print(f"Invalid choice: {selection}. Please try again.")


def _prompt_int_value(prompt: str) -> int:
    while True:
        raw = input(prompt).strip()
        _check_cancel(raw)
        try:
            return int(raw)
        except ValueError:
            print("Please enter a whole number.")


def _prompt_team_name(prompt: str) -> str:
    while True:
        name = input(prompt).strip()
        _check_cancel(name)
        if name:
            return name
        print("Please enter a team name.")


def _prompt_numeric_string(prompt: str) -> str:
    while True:
        raw = input(prompt).strip()
        _check_cancel(raw)
        if _parse_numeric(raw) is not None:
            cleaned = raw
            if cleaned.endswith("%"):
                cleaned = cleaned[:-1]
            return cleaned
        print("Please enter a numeric value.")


def _collect_manual_stats(home_team: str, away_team: str) -> Dict[str, Tuple[str, str]]:
    """Collect full stat lines keyed by canonical stat identifiers."""
    print("\n--- Manual match statistics entry ---")
    stats: Dict[str, Tuple[str, str]] = {}

    for stat_key, label in MANUAL_STAT_FIELDS:
        home_val = _prompt_numeric_string(f"{home_team} {label}: ")
        away_val = _prompt_numeric_string(f"{away_team} {label}: ")
        stats[stat_key] = (home_val, away_val)

    return stats


def _collect_manual_data() -> Tuple[str, str, Dict[str, Tuple[str, str]], int, int]:
    """Collect manual team names, score, and detailed stats."""
    print("\n--- Manual match data entry ---")
    home_team = _prompt_team_name("Enter home team name: ")
    away_team = _prompt_team_name("Enter away team name: ")
    home_score = _prompt_int_value(f"Enter {home_team} goals: ")
    away_score = _prompt_int_value(f"Enter {away_team} goals: ")
    stats = _collect_manual_stats(home_team, away_team)
    return home_team, away_team, stats, home_score, away_score


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
    try:
        input_method = _prompt_choice(
            "Choose input method (ocr/manual): ",
            {"ocr": "ocr", "o": "ocr", "manual": "manual", "m": "manual"},
        )

        if input_method == "manual":
            home_team, away_team, stats, home_score, away_score = _collect_manual_data()
        else:
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

            # Try to extract score from OCR
            score_tuple = extract_score_from_stats(stats)
            if score_tuple:
                home_score, away_score = score_tuple
                print(f"Detected score: {home_score}-{away_score}")
            else:
                # Fallback - ask for score if not found
                home_score = _prompt_int_value(f"Enter {home_team} goals: ")
                away_score = _prompt_int_value(f"Enter {away_team} goals: ")
    except InputCancelled:
        print("Input cancelled.")
        return

    print(f"\nDetected match: {home_team} vs {away_team}")
    print("Collected match stats:", list(stats.keys()))
    
    total_goals = home_score + away_score
    score = f"{home_score}-{away_score}"
    
    # MINIMAL USER INPUTS - Only what can't be determined from match stats
    
    # 1. Match setting and stage
    print("\n--- Match Context (cannot be determined from match stats) ---")
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
    
    # 2. Home/away designation - CRITICAL FIX HERE
    home_or_away = input("Is this a home, away, or neutral match for your team?: ").lower().strip()
    
    # Determine which team is "your team" (the manager's team)
    if home_or_away == "home":
        manager_team = home_team
        opponent_team = away_team
        manager_is_home = True
    elif home_or_away == "away":
        manager_team = away_team
        opponent_team = home_team
        manager_is_home = False
    else:  # neutral
        # For neutral, assume first team mentioned is manager's team
        manager_team = home_team
        opponent_team = away_team
        manager_is_home = True
    
    print(f"\nYour team: {manager_team}")
    print(f"Opponent: {opponent_team}")
    
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
    
    # PROCESS ALL OTHER DATA FROM INPUT STATS
    print("\n--- Processing match stats ---")
    
    # Extract stats with fallbacks - now checking all available stats
    def get_stat(key_variations, default_home=0, default_away=0):
        for key in key_variations:
            if key in stats:
                home_val = _parse_numeric(stats[key][0])
                away_val = _parse_numeric(stats[key][1])
                if home_val is not None and away_val is not None:
                    return home_val, away_val
        return default_home, default_away
    
    # All stats from input (these are always from home/away perspective)
    home_shots, away_shots = get_stat(['shots'], 0, 0)
    home_shots_target, away_shots_target = get_stat(['shots_on_target'], 0, 0)
    home_xg, away_xg = get_stat(['xg'], 0.0, 0.0)
    home_shots_off, away_shots_off = get_stat(['shots_off_target'], 0, 0)
    home_clear_chances, away_clear_chances = get_stat(['clear_cut_chances'], 0, 0)
    home_long_shots, away_long_shots = get_stat(['long_shots'], 0, 0)
    pos_home, pos_away = get_stat(['possession'], 50, 50)
    home_corners, away_corners = get_stat(['corners'], 0, 0)
    home_fouls, away_fouls = get_stat(['fouls'], 0, 0)
    home_offsides, away_offsides = get_stat(['offsides'], 0, 0)
    home_passes_comp, away_passes_comp = get_stat(['passes_completed'], 50, 50)
    home_crosses_comp, away_crosses_comp = get_stat(['crosses_completed'], 0, 0)
    home_tackles_won, away_tackles_won = get_stat(['tackles_won'], 50, 50)
    home_headers_won, away_headers_won = get_stat(['headers_won'], 50, 50)
    home_yellow, away_yellow = get_stat(['yellow_cards'], 0, 0)
    home_red, away_red = get_stat(['red_cards'], 0, 0)
    home_rating, away_rating = get_stat(['average_rating'], 6.5, 6.5)
    home_prog_passes, away_prog_passes = get_stat(['progressive_passes'], 0, 0)
    home_sprints, away_sprints = get_stat(['high_intensity_sprints'], 0, 0)
    
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
    
    # NOW: Remap stats to manager's team perspective
    if manager_is_home:
        # Manager is home team - stats are already correct
        manager_shots, opp_shots = home_shots, away_shots
        manager_shots_target, opp_shots_target = home_shots_target, away_shots_target
        manager_xg, opp_xg = home_xg, away_xg
        manager_shots_off, opp_shots_off = home_shots_off, away_shots_off
        manager_clear_chances, opp_clear_chances = home_clear_chances, away_clear_chances
        manager_long_shots, opp_long_shots = home_long_shots, away_long_shots
        manager_possession, opp_possession = pos_home, pos_away
        manager_corners, opp_corners = home_corners, away_corners
        manager_fouls, opp_fouls = home_fouls, away_fouls
        manager_offsides, opp_offsides = home_offsides, away_offsides
        manager_passes_comp, opp_passes_comp = home_passes_comp, away_passes_comp
        manager_crosses_comp, opp_crosses_comp = home_crosses_comp, away_crosses_comp
        manager_tackles_won, opp_tackles_won = home_tackles_won, away_tackles_won
        manager_headers_won, opp_headers_won = home_headers_won, away_headers_won
        manager_yellow, opp_yellow = home_yellow, away_yellow
        manager_red, opp_red = home_red, away_red
        manager_rating, opp_rating = home_rating, away_rating
        manager_prog_passes, opp_prog_passes = home_prog_passes, away_prog_passes
        manager_sprints, opp_sprints = home_sprints, away_sprints
        manager_score, opp_score = home_score, away_score
    else:
        # Manager is away team - SWAP ALL STATS
        manager_shots, opp_shots = away_shots, home_shots
        manager_shots_target, opp_shots_target = away_shots_target, home_shots_target
        manager_xg, opp_xg = away_xg, home_xg
        manager_shots_off, opp_shots_off = away_shots_off, home_shots_off
        manager_clear_chances, opp_clear_chances = away_clear_chances, home_clear_chances
        manager_long_shots, opp_long_shots = away_long_shots, home_long_shots
        manager_possession, opp_possession = pos_away, pos_home
        manager_corners, opp_corners = away_corners, home_corners
        manager_fouls, opp_fouls = away_fouls, home_fouls
        manager_offsides, opp_offsides = away_offsides, home_offsides
        manager_passes_comp, opp_passes_comp = away_passes_comp, home_passes_comp
        manager_crosses_comp, opp_crosses_comp = away_crosses_comp, home_crosses_comp
        manager_tackles_won, opp_tackles_won = away_tackles_won, home_tackles_won
        manager_headers_won, opp_headers_won = away_headers_won, home_headers_won
        manager_yellow, opp_yellow = away_yellow, home_yellow
        manager_red, opp_red = away_red, home_red
        manager_rating, opp_rating = away_rating, home_rating
        manager_prog_passes, opp_prog_passes = away_prog_passes, home_prog_passes
        manager_sprints, opp_sprints = away_sprints, home_sprints
        manager_score, opp_score = away_score, home_score
    
    # Display extracted stats from MANAGER'S PERSPECTIVE
    print(f"Your Team ({manager_team}) vs Opponent ({opponent_team})")
    print(f"Total Shots: {manager_shots} vs {opp_shots}")
    print(f"Shots on Target: {manager_shots_target} vs {opp_shots_target}")
    print(f"Expected Goals (xG): {manager_xg} vs {opp_xg}")
    print(f"Shots off Target: {manager_shots_off} vs {opp_shots_off}")
    print(f"Clear Cut Chances: {manager_clear_chances} vs {opp_clear_chances}")
    print(f"Long Shots: {manager_long_shots} vs {opp_long_shots}")
    print(f"Possession: {manager_possession}% vs {opp_possession}%")
    print(f"Corners: {manager_corners} vs {opp_corners}")
    print(f"Fouls: {manager_fouls} vs {opp_fouls}")
    print(f"Offsides: {manager_offsides} vs {opp_offsides}")
    print(f"Passes Completed: {manager_passes_comp}% vs {opp_passes_comp}%")
    print(f"Crosses Completed: {manager_crosses_comp}% vs {opp_crosses_comp}%")
    print(f"Tackles Won: {manager_tackles_won}% vs {opp_tackles_won}%")
    print(f"Headers Won: {manager_headers_won}% vs {opp_headers_won}%")
    print(f"Yellow Cards: {manager_yellow} vs {opp_yellow}")
    print(f"Red Cards: {manager_red} vs {opp_red}")
    print(f"Average Rating: {manager_rating} vs {opp_rating}")
    print(f"Progressive Passes: {manager_prog_passes} vs {opp_prog_passes}")
    print(f"High Intensity Sprints: {manager_sprints} vs {opp_sprints}")
    
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
    red_cards_total = manager_red + opp_red
    yellow_cards_total = manager_yellow + opp_yellow
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
    manager_aggression_score = manager_yellow + (manager_red * 3)
    opp_aggression_score = opp_yellow + (opp_red * 3)
    
    if manager_aggression_score > opp_aggression_score:
        aggr_team = manager_team
        calmer_team = opponent_team
        aggression_difference = manager_aggression_score - opp_aggression_score
    elif opp_aggression_score > manager_aggression_score:
        aggr_team = opponent_team
        calmer_team = manager_team
        aggression_difference = opp_aggression_score - manager_aggression_score
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
    manager_shot_accuracy = (manager_shots_target / manager_shots * 100) if manager_shots > 0 else 0
    opp_shot_accuracy = (opp_shots_target / opp_shots * 100) if opp_shots > 0 else 0
    
    # Calculate xG efficiency (goals vs expected goals)
    manager_xg_efficiency = (manager_score / manager_xg * 100) if manager_xg > 0 else 0
    opp_xg_efficiency = (opp_score / opp_xg * 100) if opp_xg > 0 else 0
    
    # Generate press conference questions
    print("\n--- Generating press conference questions ---")
    
    prompt = f"""
You are a football press conference reporter.

MATCH CONTEXT:
Setting: {setting.upper()}
Importance: {importance:.1f}/10 ({importance_level})
Match: {home_team} {score} {away_team} ({home_or_away})
The manager is managing {manager_team}

COMPREHENSIVE MATCH STATS ({manager_team} vs {opponent_team}):
Goals: {manager_score} vs {opp_score}
Total Shots: {manager_shots} vs {opp_shots}
Shots on Target: {manager_shots_target} vs {opp_shots_target}
Shot Accuracy: {manager_shot_accuracy:.1f}% vs {opp_shot_accuracy:.1f}%
Expected Goals (xG): {manager_xg} vs {opp_xg}
xG Efficiency: {manager_xg_efficiency:.1f}% vs {opp_xg_efficiency:.1f}%
Clear Cut Chances: {manager_clear_chances} vs {opp_clear_chances}
Long Shots: {manager_long_shots} vs {opp_long_shots}
Possession: {manager_possession}% vs {opp_possession}%
Corners: {manager_corners} vs {opp_corners}
Fouls: {manager_fouls} vs {opp_fouls}
Offsides: {manager_offsides} vs {opp_offsides}
Pass Completion: {manager_passes_comp}% vs {opp_passes_comp}%
Cross Completion: {manager_crosses_comp}% vs {opp_crosses_comp}%
Tackles Won: {manager_tackles_won}% vs {opp_tackles_won}%
Headers Won: {manager_headers_won}% vs {opp_headers_won}%
Yellow Cards: {manager_yellow} vs {opp_yellow}
Red Cards: {manager_red} vs {opp_red}
Average Rating: {manager_rating} vs {opp_rating}
Progressive Passes: {manager_prog_passes} vs {opp_prog_passes}
High Intensity Sprints: {manager_sprints} vs {opp_sprints}

GOAL SCORERS:
{manager_team}: {', '.join(home_goal_scorers if manager_is_home else away_goal_scorers) if (home_goal_scorers if manager_is_home else away_goal_scorers) else 'None'}
{opponent_team}: {', '.join(away_goal_scorers if manager_is_home else home_goal_scorers) if (away_goal_scorers if manager_is_home else home_goal_scorers) else 'None'}

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
        model="llama-3.3-70b-versatile"
    )
    
    print("\n=== PRESS CONFERENCE QUESTIONS ===")
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
