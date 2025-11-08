import pandas as pd
import numpy as np

# Load datasets
matches = pd.read_csv('matches.csv')
world_cups = pd.read_csv('world_cup.csv')
rankings = pd.read_csv('fifa_ranking.csv')

print("Matches shape before clean:", matches.shape)
print("World cups shape:", world_cups.shape)
print("Rankings shape:", rankings.shape)

# DO NOT use dropna() on matches — too many optional columns!
# Only drop duplicates
matches = matches.drop_duplicates()
world_cups = world_cups.drop_duplicates()
rankings = rankings.drop_duplicates()

# Extract year from 'Date' (safe)
matches['Date'] = pd.to_datetime(matches['Date'], errors='coerce')
matches['year'] = matches['Date'].dt.year

# Drop rows where year is missing
matches = matches.dropna(subset=['year'])
matches['year'] = matches['year'].astype(int)

# Normalize team names
replacements = {
    'Korea Republic': 'South Korea',
    'IR Iran': 'Iran',
    "Côte d'Ivoire": "Cote d'Ivoire",
    'USA': 'United States',
    'Germany FR': 'Germany',
    'West Germany': 'Germany'
}
for old, new in replacements.items():
    matches['home_team'] = matches['home_team'].replace(old, new)
    matches['away_team'] = matches['away_team'].replace(old, new)
    world_cups['Champion'] = world_cups['Champion'].replace(old, new)
    world_cups['Runner-Up'] = world_cups['Runner-Up'].replace(old, new)
    rankings['team'] = rankings['team'].replace(old, new)

# Engineer features
years = sorted(matches['year'].unique())
print("Found years:", years)

data_rows = []

for year in years:
    year_matches = matches[matches['year'] == year]
    if year_matches.empty:
        continue
    teams = set(year_matches['home_team'].unique()) | set(year_matches['away_team'].unique())
    
    # Finalists
    cup = world_cups[world_cups['Year'] == year]
    finalists = set()
    if not cup.empty:
        winner = cup['Champion'].iloc[0]
        runner = cup['Runner-Up'].iloc[0]
        finalists = {winner, runner}
    
    for team in teams:
        past_matches = matches[(matches['year'] < year) & 
                               ((matches['home_team'] == team) | (matches['away_team'] == team))]
        
        if past_matches.empty:
            win_rate = 0.0
            goal_diff_avg = 0.0
            participations = 0
        else:
            home_wins = ((past_matches['home_team'] == team) & 
                        (past_matches['home_score'] > past_matches['away_score'])).sum()
            away_wins = ((past_matches['away_team'] == team) & 
                        (past_matches['away_score'] > past_matches['home_score'])).sum()
            total = len(past_matches)
            win_rate = (home_wins + away_wins) / total if total > 0 else 0.0
            
            home_gd = past_matches[past_matches['home_team'] == team]['home_score'] - \
                      past_matches[past_matches['home_team'] == team]['away_score']
            away_gd = past_matches[past_matches['away_team'] == team]['away_score'] - \
                      past_matches[past_matches['away_team'] == team]['home_score']
            all_gd = pd.concat([home_gd, away_gd], ignore_index=True)
            goal_diff_avg = all_gd.mean() if len(all_gd) > 0 else 0.0
            
            participations = len(past_matches['year'].unique())
        
        # Ranking: Use latest rank before year (if any)
        team_ranks = rankings[rankings['team'] == team]
        if not team_ranks.empty:
            # Assume rankings are sorted by date — take last before year
            rank = team_ranks['rank'].iloc[-1]  # Simplest: latest known
        else:
            rank = 50
        
        label = 1 if team in finalists else 0
        
        data_rows.append({
            'year': year,
            'team': team,
            'win_rate': round(win_rate, 3),
            'goal_diff_avg': round(goal_diff_avg, 2),
            'participations': int(participations),
            'rank': int(rank),
            'finalist': label
        })

# Save
cleaned_data = pd.DataFrame(data_rows)
cleaned_data.to_csv('cleaned_team_data.csv', index=False)

print(f"\nSUCCESS! Created {len(cleaned_data)} rows")
print(cleaned_data.head(10))

# 2026 Teams
qualified = [
    'Argentina', 'Brazil', 'Colombia', 'Ecuador', 'Paraguay', 'Uruguay',
    'Canada', 'Mexico', 'United States',
    'Australia', 'Iran', 'Japan', 'Jordan', 'Qatar', 'Saudi Arabia', 'South Korea', 'Uzbekistan',
    'Algeria', 'Cape Verde', 'Egypt', 'Ghana', 'Morocco', 'Senegal', 'South Africa', 'Tunisia',
    'England', 'New Zealand'
]
additional = [
    'Spain', 'France', 'Portugal', 'Netherlands', 'Belgium', 'Italy', 'Germany', 'Croatia',
    'Switzerland', 'Denmark', 'Austria', 'Turkey', 'Ukraine', 'Norway', 'Sweden',
    'Poland', 'Hungary', 'Nigeria'
]
all_teams_2026 = qualified + additional
print(f"\n2026 Teams ({len(all_teams_2026)}): {all_teams_2026}")

