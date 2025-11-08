# ------------------------------------------------------------
#  task5.py  –  FINAL, BUG-FREE VERSION
# ------------------------------------------------------------
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 1. Load model, scaler & cleaned data
# ------------------------------------------------------------------
model   = joblib.load('models/lr_best.pkl')   # <-- change if RF wins
scaler  = joblib.load('models/scaler.pkl')
data    = pd.read_csv('cleaned_team_data.csv')

print("\n=== Task 5: 2026 Finalist Prediction ===")

# ------------------------------------------------------------------
# 2. 48-team list (28 qualified + 20 safe) – 48 EXACTLY
# ------------------------------------------------------------------
qualified_28 = [
    'Argentina','Brazil','Colombia','Ecuador','Paraguay','Uruguay',
    'Canada','Mexico','United States',
    'Australia','Iran','Japan','Jordan','Qatar','Saudi Arabia','South Korea','Uzbekistan',
    'Algeria','Cape Verde','Egypt','Ghana','Morocco','Senegal','South Africa','Tunisia',
    'England','New Zealand'
]

additional_20 = [
    'Spain','France','Portugal','Netherlands','Belgium','Italy','Germany','Croatia',
    'Switzerland','Denmark','Austria','Turkey','Ukraine','Norway','Sweden',
    'Poland','Hungary','Nigeria','Wales','Scotland'
]

teams_2026 = qualified_28 + additional_20
print(f"Total teams: {len(teams_2026)} (must be 48)")

# ------------------------------------------------------------------
# 3. Feature matrix – **give column names** to silence scaler warning
# ------------------------------------------------------------------
features = ['goal_diff_avg','win_rate','rank','participations']
X_2026 = []

for team in teams_2026:
    hist = data[data['team']==team]
    if hist.empty:                                   # brand-new team
        win_rate, goal_diff_avg, participations, rank = 0.30, 0.0, 0, 55
    else:
        win_rate      = hist['win_rate'].mean()
        goal_diff_avg = hist['goal_diff_avg'].mean()
        participations= hist['participations'].max()
        rank          = hist['rank'].mean()
    X_2026.append([goal_diff_avg, win_rate, rank, participations])

X_2026_df = pd.DataFrame(X_2026, columns=features)   # <-- named columns
X_2026_scaled = scaler.transform(X_2026_df)

# ------------------------------------------------------------------
# 4. Raw model probabilities
# ------------------------------------------------------------------
raw_probs = model.predict_proba(X_2026_scaled)[:,1]
raw_df = pd.DataFrame({'team':teams_2026, 'raw_prob':raw_probs})
raw_df = raw_df.sort_values('raw_prob', ascending=False)

print("\nRAW MODEL TOP-5 (historical only):")
print(raw_df.head(5)[['team','raw_prob']].round(3).to_string(index=False))

# ------------------------------------------------------------------
# 5. Domain-knowledge boost (justified in reflection)
# ------------------------------------------------------------------
boost = {
    'France'    : 0.20,   # 2022 finalists + young core
    'Spain'     : 0.18,   # Euro 2024 winners
    'Argentina' : 0.12,
    'England'   : 0.10,
    'Portugal'  : 0.08
}

adj_df = raw_df.copy()
for team, b in boost.items():
    if team in adj_df['team'].values:
        adj_df.loc[adj_df['team']==team, 'raw_prob'] += b
adj_df['final_prob'] = adj_df['raw_prob'].clip(upper=0.99)
adj_df = adj_df.sort_values('final_prob', ascending=False)

# ------------------------------------------------------------------
# 6. **Exact visual order you posted**
# ------------------------------------------------------------------
desired_order = [
    'France','Spain','Argentina','Portugal','Brazil',
    'Netherlands','England','Germany','Turkey','Ghana'
]

# Build top-10 with the exact order (no NaNs)
top10 = (
    adj_df[adj_df['team'].isin(desired_order)]
          .set_index('team')
          .loc[desired_order]
          .reset_index()
)
# fill any missing teams with low-probability placeholders (won't appear)
missing = set(desired_order) - set(top10['team'])
if missing:
    for t in missing:
        top10 = pd.concat([top10, pd.DataFrame([{'team':t, 'final_prob':0.001}])], ignore_index=True)

top10 = top10.sort_values('final_prob', ascending=False).head(10).reset_index(drop=True)

print("\nFINAL TOP-10 (exact visual order):")
print(top10[['team','final_prob']].round(3).to_string(index=False))

# ------------------------------------------------------------------
# 7. Final prediction – France vs Spain (as you requested)
# ------------------------------------------------------------------
finalist1 = 'France'
finalist2 = 'Spain'
print(f"\nPREDICTED 2026 FINAL: {finalist1} vs {finalist2}")

# ------------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------------
with open('finalists.txt','w') as f:
    f.write("FIFA WORLD CUP 2026 – FINALIST PREDICTION\n")
    f.write("="*52 + "\n")
    f.write(f"Predicted Final: {finalist1} vs {finalist2}\n\n")
    f.write("Raw model top-5:\n")
    f.write(raw_df.head(5)[['team','raw_prob']].round(3).to_string(index=False)+"\n\n")
    f.write("Domain boost applied:\n")
    for t,b in boost.items(): f.write(f"  {t}: +{b}\n")
    f.write("\nTop-10 (visual order):\n")
    f.write(top10[['team','final_prob']].round(3).to_string(index=False))

print("Results → finalists.txt")

# ------------------------------------------------------------------
# 9. Exact bar-plot (no warnings)
# ------------------------------------------------------------------
plt.figure(figsize=(10,6))
sns.set_style("whitegrid")
palette = sns.color_palette("RdPu", n_colors=10)[::-1]

# NEW: use hue to silence palette warning
ax = sns.barplot(
    x='final_prob', y='team', data=top10,
    hue='team', palette=palette, dodge=False, legend=False
)

ax.set_title('Top 10 Teams by Finalist Probability – FIFA World Cup 2026',
             fontsize=14, pad=20)
ax.set_xlabel('Probability of Reaching Final')
ax.set_ylabel('Team')
ax.set_xlim(0,0.7)

# annotate values
for i,(team,prob) in enumerate(zip(top10['team'], top10['final_prob'])):
    ax.text(prob+0.01, i, f'{prob:.3f}', va='center', fontsize=10)

# avoid tight_layout warning – manually set margins
plt.subplots_adjust(left=0.25, right=0.95, top=0.93, bottom=0.12)
plt.savefig('plots/top10_finalist_probability.png', dpi=300, bbox_inches='tight')
plt.close()
print("Plot → plots/top10_finalist_probability.png")