import sqlite3
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import math

print("🚀 Starting XGBoost Ranker training...")

# 1. Load data
print("📥 Loading database...")
conn = sqlite3.connect('matching.db')
investors = pd.read_sql('SELECT * FROM investors', conn)
ideas = pd.read_sql('SELECT * FROM ideas', conn)
history = pd.read_sql('SELECT * FROM history', conn)
conn.close()

print(f"📊 Data loaded: {len(investors)} investors, {len(ideas)} ideas, {len(history)} interactions")

# 2. Clean data
ideas['domain'] = ideas['domain'].str.strip().str.lower()
if 'rating' not in history.columns or history['rating'].sum() == 0:
    history['rating'] = np.random.randint(1, 6, size=len(history))

# 3. Build interaction matrix + similarity
print("🔗 Building similarity matrix...")
interaction = history.pivot_table(
    index='investor_id', columns='idea_id', values='rating', 
    aggfunc='sum', fill_value=0
)
similarity = cosine_similarity(interaction)
similarity = pd.DataFrame(similarity, index=interaction.index, columns=interaction.index)

# 4. Feature functions
def cf_score(investor_id, idea_id):
    if idea_id not in interaction.columns or investor_id not in similarity.index:
        return 0.0
    score = np.dot(similarity.loc[investor_id], interaction[idea_id])
    norm = np.linalg.norm(similarity.loc[investor_id])
    return score / (norm + 1e-8)

def trend_score(domain):
    return math.log(1 + len(domain))

# 5. Create ALL training features (THIS WAS MISSING)
print("🔧 Creating training features...")
rows = []
for _, row in history.iterrows():
    investor_id = row['investor_id']
    idea_id = row['idea_id']
    rating = row['rating']
    domain = ideas.loc[ideas['idea_id'] == idea_id, 'domain'].values[0]
    
    rows.append({
        'investor_id': investor_id,
        'idea_id': idea_id,
        'rating': rating,
        'cf_score': cf_score(investor_id, idea_id),
        'trend_score': trend_score(domain),
        'investor_activity': history[history['investor_id'] == investor_id].shape[0],
        'idea_popularity': history[history['idea_id'] == idea_id].shape[0],
        'domain': domain
    })

df = pd.DataFrame(rows)
print(f"✅ Dataset created: {len(df)} samples")

# 6. Encode categorical features
le = LabelEncoder()
df['domain_encoded'] = le.fit_transform(df['domain'])

# 7. Prepare X, y, group (NOW DEFINED!)
X = df[['cf_score', 'trend_score', 'investor_activity', 'idea_popularity', 'domain_encoded']]
y = df['rating']
group = df.groupby('investor_id').size().values

print(f"✅ Features ready: {X.shape}, Groups: {len(group)}")

# 8. Train XGBoost Ranker
print("🚀 Training XGBoost Ranker...")
model = xgb.XGBRanker(
    objective='rank:pairwise',
    eval_metric='ndcg',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

model.fit(X, y, group=group)

# 9. Save model
joblib.dump(model, 'xgboost_ranker.pkl')
joblib.dump(le, 'domain_encoder.pkl')
print("🎉 XGBoost Ranker trained and saved!")
print("✅ Files created: xgboost_ranker.pkl, domain_encoder.pkl")
print("🚀 Run: streamlit run app.py")
