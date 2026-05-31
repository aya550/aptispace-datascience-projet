import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json

st.set_page_config(page_title="AptiSpace", layout="wide", page_icon="🎧")

st.markdown("""
    <style>
    .stApp { background-color: #0b0c10; color: #c5c6c7; font-family: 'Inter', sans-serif; }
    h1, h2, h3 { color: #66fcf1; letter-spacing: 1px; font-weight: 600; }
    .story-text { font-size: 1.05rem; line-height: 1.7; color: #e0e2e4; margin-bottom: 1.5rem;
                  background: #1f2833; padding: 18px 22px; border-radius: 10px;
                  border-left: 5px solid #45a29e; }
    .stat-box { background: #1f2833; padding: 20px; border-radius: 10px; text-align: center;
                border: 1px solid #45a29e; box-shadow: 0 4px 15px rgba(102,252,241,0.1); }
    .stat-box h2 { font-size: 2.5rem; margin: 0; color: #66fcf1; }
    .stat-box p  { margin: 0; font-size: 0.9rem; color: #c5c6c7; text-transform: uppercase;
                   letter-spacing: 1px; }
    .anomaly-card { background: #1f2833; padding: 12px 16px; border-radius: 8px;
                    margin-bottom: 8px; border-left: 4px solid #45a29e; }
    </style>
""", unsafe_allow_html=True)

# ── Chargement ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_artifacts():
    rf      = joblib.load('data/processed/rf_unified_model.pkl')
    scaler  = joblib.load('data/processed/audio_scaler.pkl')
    with open('data/processed/model_artifacts.json', 'r') as f:
        artifacts = json.load(f)
    genre_centroids = pd.DataFrame(artifacts['genre_centroids'])
    return rf, scaler, artifacts, genre_centroids

@st.cache_data
def build_full_analysis(_scaler, _rf, _genre_centroids, _artifacts):
    """Pipeline complet : feature engineering + prédictions + résidus sur tout le dataset."""
    AUDIO = _artifacts['audio_features']
    df = pd.read_csv('data/processed/cleaned_data_sample.csv')

    df['artist_reputation'] = df.groupby('artists')['popularity'].transform('mean')
    df['pop_rank_in_genre'] = df.groupby('track_genre')['popularity'].transform(
        lambda x: x.rank(pct=True)
    )
    audio_scaled    = _scaler.transform(df[AUDIO])
    centroid_matrix = np.array([_genre_centroids.loc[g].values for g in df['track_genre']])
    df['genre_distance'] = np.linalg.norm(audio_scaled - centroid_matrix, axis=1)
    df['duration_min']   = df['duration_ms'] / 60000
    df['explicit_int']   = df['explicit'].astype(int)

    X = df[_artifacts['model_features']]
    df['predicted_rank'] = _rf.predict(X)
    df['residual']       = df['pop_rank_in_genre'] - df['predicted_rank']
    return df

try:
    rf, scaler, artifacts, genre_centroids = load_model_artifacts()
    model_ok = True
except Exception as e:
    st.error(f"Impossible de charger le modèle : {e}")
    model_ok = False
    st.stop()

AUDIO_FEATURES = artifacts['audio_features']
GENRES_LIST    = artifacts['genres_list']

with st.spinner("Chargement de l'analyse complète…"):
    df_full = build_full_analysis(scaler, rf, genre_centroids, artifacts)

# ══════════════════════════════════════════════════════════════════════════════
# EN-TÊTE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h1>🎧 AptiSpace : Prédire le Succès Musical</h1>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.markdown("<div class='stat-box'><p>Pistes analysées</p><h2>97 270</h2></div>", unsafe_allow_html=True)
c2.markdown("<div class='stat-box'><p>Genres</p><h2>114</h2></div>", unsafe_allow_html=True)
c3.markdown("<div class='stat-box'><p>R² Modèle Unifié</p><h2>0.31</h2></div>", unsafe_allow_html=True)
c4.markdown("<div class='stat-box'><p>Features clés</p><h2>13</h2></div>", unsafe_allow_html=True)

st.write("")
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHAPITRE 1 — POURQUOI L'AUDIO SEUL NE SUFFIT PAS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h2>📉 Chapitre 1 : L'Audio ne Prédit pas le Succès</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; margin-bottom:1.2rem;'>Aucune feature audio ne dépasse <b>r = 0.07</b> avec la popularité. Le son seul ne suffit pas.</p>", unsafe_allow_html=True)

corr_vals = df_full[AUDIO_FEATURES].corrwith(df_full['popularity']).reset_index()
corr_vals.columns = ['Feature', 'r']
corr_vals = corr_vals.reindex(corr_vals['r'].abs().sort_values().index)
corr_vals['couleur'] = corr_vals['r'].apply(lambda v: '#1DB954' if v > 0 else '#ff4500')
corr_vals['label']   = corr_vals['Feature'].map({
    'danceability': 'Dansabilité', 'energy': 'Énergie', 'loudness': 'Volume',
    'speechiness': 'Paroles', 'acousticness': 'Acoustique',
    'instrumentalness': 'Instrumental', 'liveness': 'Live',
    'valence': 'Valence', 'tempo': 'Tempo'
}).fillna(corr_vals['Feature'])

fig_corr = go.Figure(go.Bar(
    x=corr_vals['r'], y=corr_vals['label'], orientation='h',
    marker_color=corr_vals['couleur'],
    text=corr_vals['r'].round(3), textposition='outside'
))
fig_corr.add_vline(x=0, line_width=1, line_color='white', opacity=0.3)
fig_corr.add_annotation(x=-0.17, y=8.6, text="Pas de paroles = moins populaire",
    showarrow=False, font=dict(color='#ff4500', size=10))
fig_corr.add_annotation(x=0.055, y=0.4, text="Max r = 0.07 — quasi nul",
    showarrow=False, font=dict(color='#1DB954', size=10))
fig_corr.update_layout(
    title="Corrélation Features Audio → Popularité  (r de Pearson)",
    xaxis=dict(range=[-0.25, 0.20], showgrid=False, zeroline=False, title="r"),
    yaxis=dict(showgrid=False),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#141b22',
    font_color='#c5c6c7', height=360,
    margin=dict(t=50, b=20, l=10, r=90)
)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHAPITRE 2 — CE QUI PRÉDIT VRAIMENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h2>🎯 Chapitre 2 : Ce qui Prédit Vraiment le Succès</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; margin-bottom:1.2rem;'>La <b style='color:#1DB954;'>réputation de l'artiste ★</b> écrase toutes les features audio — à elle seule, elle représente ~67% du pouvoir prédictif.</p>", unsafe_allow_html=True)

col2a, col2b = st.columns([3, 2])

imp_df = pd.DataFrame({
    'feature':    artifacts['model_features'],
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=True)

imp_df['label'] = imp_df['feature'].map({
    'artist_reputation': 'Réputation artiste ★',
    'genre_distance':    'Distance au genre ★',
    'duration_min':      'Durée (min)',
    'explicit_int':      'Contenu explicite',
    'danceability':      'Dansabilité',
    'energy':            'Énergie',
    'loudness':          'Volume',
    'speechiness':       'Paroles',
    'acousticness':      'Acoustique',
    'instrumentalness':  'Instrumental',
    'liveness':          'Live',
    'valence':           'Valence',
    'tempo':             'Tempo',
}).fillna(imp_df['feature'])

imp_df['couleur'] = imp_df['feature'].apply(
    lambda f: '#1DB954' if f in ['artist_reputation', 'genre_distance'] else '#2a4a5a'
)

with col2a:
    fig_imp = go.Figure(go.Bar(
        x=imp_df['importance'], y=imp_df['label'], orientation='h',
        marker_color=imp_df['couleur'],
        text=(imp_df['importance'] * 100).round(1).astype(str) + '%',
        textposition='outside'
    ))
    fig_imp.update_layout(
        title="Importance des Variables",
        xaxis=dict(showgrid=False, zeroline=False, title="Importance (Gini)"),
        yaxis=dict(showgrid=False),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#141b22',
        font_color='#c5c6c7', height=420,
        margin=dict(t=40, b=20, l=10, r=80)
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with col2b:
    df_scatter = df_full.sample(min(3000, len(df_full)), random_state=1)
    x_vals = df_scatter['artist_reputation'].values
    y_vals = df_scatter['pop_rank_in_genre'].values
    m, b   = np.polyfit(x_vals, y_vals, 1)
    x_line = np.array([x_vals.min(), x_vals.max()])

    fig_scat = go.Figure()
    fig_scat.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode='markers',
        marker=dict(color='#1DB954', opacity=0.2, size=4),
        hoverinfo='skip', showlegend=False
    ))
    fig_scat.add_trace(go.Scatter(
        x=x_line, y=m * x_line + b, mode='lines',
        line=dict(color='#ff007f', width=2.5), showlegend=False
    ))
    fig_scat.update_layout(
        title="Réputation Artiste → Rang dans le Genre",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#141b22',
        font_color='#c5c6c7', height=420, showlegend=False,
        xaxis=dict(title='Réputation artiste (0–100)', showgrid=False, zeroline=False),
        yaxis=dict(title='Rang relatif (0–1)', showgrid=False, zeroline=False),
        margin=dict(t=40, b=40, l=50, r=20)
    )
    st.plotly_chart(fig_scat, use_container_width=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHAPITRE 3 — LES ANOMALIES DE SUCCÈS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h2>🎤 Chapitre 3 : Sur Scène — Les Anomalies</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; margin-bottom:1.2rem;'>Les points <b style='color:#1DB954;'>verts ★</b> ont explosé sans que le modèle le voie venir. Les <b style='color:#ff4500;'>rouges ✕</b> auraient dû cartonner — ils ont raté. Survolez les points pour voir les titres.</p>", unsafe_allow_html=True)

col_filt1, _ = st.columns([1, 3])
with col_filt1:
    genre_filter = st.selectbox("Genre", ["Tous"] + sorted(df_full['track_genre'].unique().tolist()), index=0, label_visibility="collapsed")

df_anom = df_full.copy() if genre_filter == "Tous" else df_full[df_full['track_genre'] == genre_filter].copy()
df_anom = df_anom.rename(columns={'pop_rank_in_genre': 'actual_rank'})

seuil_sup = df_anom['residual'].quantile(0.95)
seuil_inf = df_anom['residual'].quantile(0.05)

# ── Scatter "La Scène" ──────────────────────────────────────────────────────
_bg_pool = df_anom[(df_anom['residual'] > seuil_inf) & (df_anom['residual'] < seuil_sup)]
df_bg = _bg_pool.sample(min(2500, len(_bg_pool)), random_state=42)
top_surp_pts = df_anom[df_anom['residual'] >= seuil_sup]
top_dec_pts  = df_anom[df_anom['residual'] <= seuil_inf]
labeled_surp = df_anom.nlargest(10, 'residual')
labeled_dec  = df_anom.nsmallest(10, 'residual')

fig_stage = go.Figure()

# Lignes de portée musicale (fond)
for y in [0.2, 0.4, 0.6, 0.8]:
    fig_stage.add_hline(y=y, line_width=1, line_color='rgba(69,162,158,0.15)')

# Diagonale "prédiction parfaite"
fig_stage.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode='lines',
    line=dict(color='rgba(255,255,255,0.18)', width=1.5, dash='dot'),
    showlegend=False, hoverinfo='skip'
))

# Fond (morceaux normaux)
fig_stage.add_trace(go.Scatter(
    x=df_bg['predicted_rank'], y=df_bg['actual_rank'], mode='markers',
    marker=dict(color='#1f2e3a', size=4, opacity=0.7),
    showlegend=False, hoverinfo='skip'
))

# Surprises
fig_stage.add_trace(go.Scatter(
    x=top_surp_pts['predicted_rank'], y=top_surp_pts['actual_rank'], mode='markers',
    marker=dict(color='#1DB954', size=8, opacity=0.75, line=dict(color='#0d8a3a', width=1)),
    name='Surprises', hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Genre : %{customdata[2]}<br>Pop : %{customdata[3]}<extra></extra>',
    customdata=np.stack([top_surp_pts['track_name'], top_surp_pts['artists'], top_surp_pts['track_genre'], top_surp_pts['popularity'].astype(int)], axis=-1)
))

# Déceptions
fig_stage.add_trace(go.Scatter(
    x=top_dec_pts['predicted_rank'], y=top_dec_pts['actual_rank'], mode='markers',
    marker=dict(color='#ff4500', size=8, opacity=0.75, line=dict(color='#cc3700', width=1)),
    name='Déceptions', hovertemplate='<b>%{customdata[0]}</b><br>%{customdata[1]}<br>Genre : %{customdata[2]}<br>Pop : %{customdata[3]}<extra></extra>',
    customdata=np.stack([top_dec_pts['track_name'], top_dec_pts['artists'], top_dec_pts['track_genre'], top_dec_pts['popularity'].astype(int)], axis=-1)
))

# Labels top 10 surprises (étoiles)
fig_stage.add_trace(go.Scatter(
    x=labeled_surp['predicted_rank'], y=labeled_surp['actual_rank'],
    mode='markers+text',
    marker=dict(color='#1DB954', size=14, symbol='star', line=dict(color='white', width=1)),
    text=labeled_surp['track_name'].str[:22],
    textposition='top center', textfont=dict(size=8, color='#1DB954'),
    showlegend=False, hoverinfo='skip'
))

# Labels top 10 déceptions (croix)
fig_stage.add_trace(go.Scatter(
    x=labeled_dec['predicted_rank'], y=labeled_dec['actual_rank'],
    mode='markers+text',
    marker=dict(color='#ff4500', size=14, symbol='x', line=dict(color='white', width=1)),
    text=labeled_dec['track_name'].str[:22],
    textposition='bottom center', textfont=dict(size=8, color='#ff4500'),
    showlegend=False, hoverinfo='skip'
))

fig_stage.update_layout(
    xaxis=dict(title="Rang prédit →", showgrid=False, zeroline=False,
               range=[-0.05, 1.05], title_font=dict(color='#888')),
    yaxis=dict(title="↑ Rang réel", showgrid=False, zeroline=False,
               range=[-0.05, 1.12], title_font=dict(color='#888')),
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#080d12',
    font_color='#c5c6c7', height=520,
    legend=dict(font=dict(color='#c5c6c7'), bgcolor='rgba(0,0,0,0)',
                orientation='h', y=1.04, x=0),
    annotations=[
        dict(x=0.12, y=0.93, text="🔥 zone surprises", showarrow=False,
             font=dict(color='#1DB954', size=11), bgcolor='rgba(0,0,0,0.4)', borderpad=4),
        dict(x=0.88, y=0.08, text="🧊 zone déceptions", showarrow=False,
             font=dict(color='#ff4500', size=11), bgcolor='rgba(0,0,0,0.4)', borderpad=4),
    ],
    margin=dict(t=40, b=40, l=50, r=30)
)
st.plotly_chart(fig_stage, use_container_width=True)

# ── Cartes setlist ──────────────────────────────────────────────────────────
col3a, col3b = st.columns(2)

def render_cards(col, tracks, color, arrow):
    with col:
        st.markdown(
            f"<h4 style='color:{color}; letter-spacing:1px;'>"
            f"{'🔥 Hits Inattendus' if arrow == '▲' else '🧊 Déceptions'}</h4>",
            unsafe_allow_html=True
        )
        for _, row in tracks.iterrows():
            name   = str(row['track_name'])[:38] + ('…' if len(str(row['track_name'])) > 38 else '')
            artist = str(row['artists']).split(';')[0][:32]
            res    = row['residual']
            bar_w  = int(abs(res) * 100)
            st.markdown(f"""
<div style='background:linear-gradient(135deg,
    {"#0a1f0a" if arrow=="▲" else "#1f0a0a"} 0%, #0b0c10 100%);
    border-left:3px solid {color};
    border-radius:8px; padding:10px 14px; margin-bottom:7px;'>
  <span style='color:{color}; font-size:0.65rem; font-weight:700;
               letter-spacing:2px; text-transform:uppercase;'>
    {arrow} {res:+.2f}
  </span>
  <div style='color:#fff; font-weight:700; font-size:0.95rem; margin:5px 0 2px; line-height:1.3;'>
    {name}
  </div>
  <div style='color:{color}; font-size:0.82rem; font-weight:600; margin-bottom:2px;'>
    {artist}
  </div>
  <div style='color:#555; font-size:0.75rem;'>
    <span style='color:#45a29e;'>{row['track_genre']}</span>
  </div>
  <div style='background:{color}33; height:3px; border-radius:2px;
              width:{bar_w}%; margin-top:8px; max-width:100%;'></div>
</div>""", unsafe_allow_html=True)

render_cards(col3a, df_anom.nlargest(5, 'residual'), '#1DB954', '▲')
render_cards(col3b, df_anom.nsmallest(5, 'residual'), '#ff4500', '▼')

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CHAPITRE 4 — STUDIO VIRTUEL
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<h2>🎸 Chapitre 4 : Studio Virtuel</h2>", unsafe_allow_html=True)
st.markdown("<p style='color:#888; margin-bottom:1.5rem;'>Composez un morceau fictif. L'égaliseur se met à jour en direct. Cliquez <b>Analyser</b> pour voir votre empreinte sonore face au genre.</p>", unsafe_allow_html=True)

col_studio, col_eq = st.columns([1, 1])

with col_studio:
    selected_genre = st.selectbox("Genre", GENRES_LIST,
        index=GENRES_LIST.index('pop') if 'pop' in GENRES_LIST else 0)
    artist_rep = st.slider("Réputation artiste  (0 = inconnu  ·  100 = megastar)",
                           0.0, 100.0, 40.0, 1.0)

    c1, c2 = st.columns(2)
    with c1:
        dance    = st.slider("Dansabilité",  0.0, 1.0, 0.65, 0.01)
        energy   = st.slider("Énergie",      0.0, 1.0, 0.75, 0.01)
        loudness = st.slider("Volume (dB)", -60.0, 0.0, -5.0, 0.5)
        acoustic = st.slider("Acoustique",   0.0, 1.0, 0.10, 0.01)
    with c2:
        instr    = st.slider("Instrumental", 0.0, 1.0, 0.00, 0.01)
        speech   = st.slider("Paroles",      0.0, 1.0, 0.05, 0.01)
        live     = st.slider("Liveness",     0.0, 1.0, 0.10, 0.01)
        valence  = st.slider("Valence",      0.0, 1.0, 0.60, 0.01)

    tempo        = st.slider("Tempo (BPM)", 40, 200, 120, 1)
    duration_min = st.slider("Durée (min)",  0.5, 10.0, 3.5, 0.1)
    explicit_int = int(st.checkbox("Contenu explicite"))

with col_eq:
    # ── Égaliseur temps réel ────────────────────────────────────────────────
    eq_labels = ['Danse', 'Énergie', 'Volume', 'Acoust.', 'Instru.', 'Paroles', 'Live', 'Valence']
    eq_raw    = [dance, energy, (loudness + 60) / 60, acoustic, instr, speech, live, valence]

    fig_eq = go.Figure()
    fig_eq.add_trace(go.Bar(
        x=eq_labels, y=eq_raw,
        marker=dict(color='#1DB954', line=dict(color='#0a2a0a', width=1)),
        showlegend=False, width=0.6
    ))
    # Marqueurs peak
    for lbl, val in zip(eq_labels, eq_raw):
        fig_eq.add_trace(go.Scatter(
            x=[lbl], y=[min(val + 0.03, 1.0)],
            mode='markers',
            marker=dict(color='white', size=7, symbol='line-ew',
                        line=dict(color='white', width=2)),
            showlegend=False
        ))
    for y in [0.25, 0.5, 0.75, 1.0]:
        fig_eq.add_hline(y=y, line_width=1, line_color='rgba(255,255,255,0.06)')

    fig_eq.update_layout(
        title=dict(text="Égaliseur — profil audio du morceau",
                   font=dict(color='#555', size=11), x=0.5),
        plot_bgcolor='#050a05', paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 1.12], showgrid=False, zeroline=False,
                   tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                   ticktext=['0', '0.25', '0.5', '0.75', '1'],
                   tickfont=dict(color='#333', size=9)),
        xaxis=dict(showgrid=False, tickfont=dict(color='#1DB954', size=10)),
        height=280, margin=dict(t=30, b=10, l=30, r=10)
    )
    st.plotly_chart(fig_eq, use_container_width=True)

    # ── Compteur BPM style affichage numérique ──────────────────────────────
    bpm_color = '#1DB954' if tempo < 100 else '#ffd700' if tempo < 140 else '#ff4500'
    st.markdown(f"""
<div style='background:#050a05; border:1px solid #1f2833; border-radius:8px;
            padding:14px; text-align:center; margin-top:4px;'>
  <div style='color:#333; font-size:0.7rem; letter-spacing:3px; text-transform:uppercase;'>BPM</div>
  <div style='color:{bpm_color}; font-size:3rem; font-weight:900; font-family:monospace;
              line-height:1; text-shadow: 0 0 20px {bpm_color}88;'>
    {tempo:03d}
  </div>
  <div style='color:#333; font-size:0.65rem; margin-top:4px;'>
    {"SLOW" if tempo < 90 else "MID" if tempo < 130 else "FAST" if tempo < 160 else "HYPER"}
  </div>
</div>""", unsafe_allow_html=True)

    # ── Réputation artiste — jauge circulaire compacte ──────────────────────
    fig_rep = go.Figure(go.Indicator(
        mode="gauge+number",
        value=artist_rep,
        number={'font': {'color': '#66fcf1', 'size': 22}, 'suffix': '/100'},
        title={'text': "Réputation Artiste", 'font': {'color': '#666', 'size': 11}},
        gauge={
            'axis': {'range': [0, 100], 'visible': False},
            'bar':  {'color': '#66fcf1', 'thickness': 0.25},
            'bgcolor': '#0d1117',
            'steps': [
                {'range': [0,  33], 'color': '#0d1117'},
                {'range': [33, 66], 'color': '#0f1f1f'},
                {'range': [66, 100],'color': '#0f2a1f'},
            ],
        }
    ))
    fig_rep.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', height=130,
        margin=dict(t=30, b=0, l=20, r=20)
    )
    st.plotly_chart(fig_rep, use_container_width=True)

st.write("")
if st.button("🎵  ANALYSER CE MORCEAU", use_container_width=True, type="primary"):

    audio_input        = pd.DataFrame(
        [[dance, energy, loudness, speech, acoustic, instr, live, valence, tempo]],
        columns=AUDIO_FEATURES
    )
    audio_scaled_input = scaler.transform(audio_input).flatten()

    if selected_genre in genre_centroids.index:
        centroid = genre_centroids.loc[selected_genre].values
        g_dist   = float(np.linalg.norm(audio_scaled_input - centroid))
    else:
        g_dist   = float(np.linalg.norm(audio_scaled_input - genre_centroids.mean(axis=0).values))

    X_new = pd.DataFrame([[
        dance, energy, loudness, speech, acoustic, instr,
        live, valence, tempo, artist_rep, g_dist, duration_min, explicit_int
    ]], columns=artifacts['model_features'])

    predicted_rank = float(rf.predict(X_new)[0])
    predicted_rank = max(0.0, min(1.0, predicted_rank))
    percentile_top = round((1 - predicted_rank) * 100, 1)

    if   predicted_rank >= 0.75: result_color, verdict = '#1DB954', f"🔥 TOP {percentile_top}% — Fort potentiel !"
    elif predicted_rank >= 0.50: result_color, verdict = '#ffd700', f"🎵 TOP {percentile_top}% — Au-dessus de la médiane"
    elif predicted_rank >= 0.25: result_color, verdict = '#ff8c00', f"📉 TOP {percentile_top}% — En dessous de la médiane"
    else:                         result_color, verdict = '#ff4500', f"🧊 TOP {percentile_top}% — Difficile de percer"

    st.markdown("<hr style='border-color:#1f2833;'>", unsafe_allow_html=True)
    col_res1, col_res2 = st.columns([1, 1])

    # ── Vinyl / Gauge ───────────────────────────────────────────────────────
    with col_res1:
        score = round(predicted_rank * 100, 1)
        # Disque vinyle en polar + score au centre
        angles = list(range(0, 360, 3))
        fig_vinyl = go.Figure()
        # Sillons
        for r_sillon in [20, 35, 50, 65, 80]:
            fig_vinyl.add_trace(go.Scatterpolar(
                r=[r_sillon]*len(angles), theta=angles, mode='lines',
                line=dict(color='#1a1a1a', width=1), showlegend=False, hoverinfo='skip'
            ))
        # Arc de score (couleur)
        arc_theta = list(range(0, int(score * 3.6)))
        arc_r     = [72] * len(arc_theta)
        fig_vinyl.add_trace(go.Scatterpolar(
            r=arc_r, theta=arc_theta, mode='lines',
            line=dict(color=result_color, width=8), showlegend=False, hoverinfo='skip'
        ))
        # Centre (label)
        fig_vinyl.add_trace(go.Scatterpolar(
            r=[0], theta=[0], mode='markers+text',
            marker=dict(size=1, color='rgba(0,0,0,0)'),
            text=[f"<b>{score:.0f}</b>"], textfont=dict(color=result_color, size=26),
            showlegend=False
        ))
        fig_vinyl.update_layout(
            polar=dict(
                bgcolor='#0a0a0a',
                angularaxis=dict(visible=False),
                radialaxis=dict(visible=False, range=[0, 90])
            ),
            paper_bgcolor='rgba(0,0,0,0)', height=280,
            margin=dict(t=20, b=20, l=20, r=20),
            annotations=[dict(
                x=0.5, y=0.38, text=verdict,
                showarrow=False, font=dict(color=result_color, size=13),
                xref='paper', yref='paper'
            )]
        )
        st.plotly_chart(fig_vinyl, use_container_width=True)

    # ── Radar empreinte sonore ──────────────────────────────────────────────
    with col_res2:
        radar_feats  = ['danceability','energy','speechiness','acousticness',
                        'instrumentalness','liveness','valence']
        radar_labels = ['Danse','Énergie','Paroles','Acoustique','Instru.','Live','Valence']
        user_vals    = [dance, energy, speech, acoustic, instr, live, valence]

        if selected_genre in df_full['track_genre'].values:
            genre_avg = df_full[df_full['track_genre'] == selected_genre][radar_feats].mean().tolist()
        else:
            genre_avg = df_full[radar_feats].mean().tolist()

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=genre_avg + [genre_avg[0]],
            theta=radar_labels + [radar_labels[0]],
            fill='toself', fillcolor='rgba(69,162,158,0.15)',
            line=dict(color='#45a29e', width=2, dash='dot'),
            name=f'Moy. {selected_genre}'
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=user_vals + [user_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill='toself', fillcolor=f'rgba(29,185,84,0.2)',
            line=dict(color=result_color, width=2.5),
            name='Ton morceau'
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor='#0a0f0a',
                angularaxis=dict(tickfont=dict(color='#c5c6c7', size=10), linecolor='#1f2833'),
                radialaxis=dict(range=[0, 1], showticklabels=False, gridcolor='#1f2833',
                                linecolor='#1f2833')
            ),
            paper_bgcolor='rgba(0,0,0,0)', font_color='#c5c6c7',
            legend=dict(font=dict(size=10), bgcolor='rgba(0,0,0,0)', x=0.3, y=-0.1,
                        orientation='h'),
            height=310, margin=dict(t=20, b=40, l=30, r=30),
            title=dict(text="Empreinte Sonore vs Profil du Genre",
                       font=dict(color='#666', size=11), x=0.5)
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Interprétation ──────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#1f2833; margin:8px 0 16px;'>", unsafe_allow_html=True)

    typ_label = (
        "très typique du genre — le morceau respecte les codes sonores attendus"
        if g_dist < 1.5 else
        "atypique / innovant — il s'écarte du profil standard du genre"
        if g_dist > 3.0 else
        "modérément typique — quelques originalités sans trop s'éloigner"
    )

    # Différences radar : quelles features s'écartent le plus de la moyenne du genre
    diffs = [(lab, uv - gv) for lab, uv, gv in zip(radar_labels, user_vals, genre_avg)]
    above = [(l, d) for l, d in diffs if d > 0.15]
    below = [(l, d) for l, d in diffs if d < -0.15]

    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        st.markdown(f"""
<div style='background:#0d1117; border:1px solid {result_color}44;
            border-left:3px solid {result_color}; border-radius:8px; padding:14px;'>
  <div style='color:#555; font-size:0.7rem; text-transform:uppercase; letter-spacing:2px;'>Rang dans le genre</div>
  <div style='color:{result_color}; font-size:1.6rem; font-weight:700; margin:6px 0;'>{verdict}</div>
  <div style='color:#666; font-size:0.8rem;'>Score <b style='color:#c5c6c7;'>{score:.0f}/100</b> parmi tous les morceaux de <b style='color:#c5c6c7;'>{selected_genre}</b></div>
</div>""", unsafe_allow_html=True)

    with col_i2:
        above_txt = ", ".join([f"<b>{l}</b> (+{d:.2f})" for l, d in above]) if above else "—"
        below_txt = ", ".join([f"<b>{l}</b> ({d:.2f})" for l, d in below]) if below else "—"
        above_line = f"<div style='margin-top:6px;'>↑ Plus fort : {above_txt}</div>" if above else ""
        below_line = f"<div style='margin-top:4px;'>↓ Plus faible : {below_txt}</div>" if below else ""
        st.markdown(f"""
<div style='background:#0d1117; border:1px solid #1f2833;
            border-radius:8px; padding:14px; height:100%;'>
  <div style='color:#555; font-size:0.7rem; text-transform:uppercase; letter-spacing:2px;'>Empreinte sonore</div>
  <div style='color:#c5c6c7; font-size:0.8rem; margin-top:8px;'>
    Profil <b style='color:#66fcf1;'>{typ_label.split("—")[0].strip()}</b>
    {above_line}{below_line}
  </div>
</div>""", unsafe_allow_html=True)

    with col_i3:
        rep_label = (
            "Artiste peu connu — la réputation pèse lourd dans la prédiction" if artist_rep < 30
            else "Artiste établi — bon levier de popularité" if artist_rep < 70
            else "Megastar — réputation très favorable au succès"
        )
        st.markdown(f"""
<div style='background:#0d1117; border:1px solid #1f2833;
            border-radius:8px; padding:14px; height:100%;'>
  <div style='color:#555; font-size:0.7rem; text-transform:uppercase; letter-spacing:2px;'>Contexte artiste</div>
  <div style='color:#66fcf1; font-size:1.4rem; font-weight:700; margin:6px 0;'>{artist_rep:.0f} / 100</div>
  <div style='color:#666; font-size:0.8rem;'>{rep_label}</div>
</div>""", unsafe_allow_html=True)
