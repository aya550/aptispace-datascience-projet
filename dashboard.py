import pandas as pd
import numpy as np
import sys, os
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
sys.path.append(os.path.abspath('.'))
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc


df = pd.read_csv('data/processed/cleaned_data_sample.csv')
df['duration_min'] = df['duration_ms'] / 60000
df_feat = df.copy()
df_feat['is_popular'] = (df_feat['popularity'] >= 50).astype(int)
df_feat['dance_energy_score'] = df_feat['danceability'] * df_feat['energy']
df_feat['explicit_int'] = df_feat['explicit'].astype(int)

# visualisation

# Graphique 1 — Top 15 genres
top15 = df_feat.groupby('track_genre')['popularity'].mean()\
               .sort_values(ascending=False).head(15).reset_index()
fig_top15 = px.bar(
    top15, x='track_genre', y='popularity',
    color='popularity', color_continuous_scale='Greens',
    title="Top 15 des genres les plus populaires",
    labels={'popularity': 'Popularite moyenne', 'track_genre': 'Genre'},
    text=top15['popularity'].round(1)
)
fig_top15.update_layout(
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white', showlegend=False,
    xaxis_tickangle=30
)

# Graphique 2 — Nuage de points (top 5 genres)
df_sample = df_feat.sample(2000, random_state=42)
top5_genres = df_feat.groupby('track_genre')['popularity'].mean()\
                     .sort_values(ascending=False).head(5).index
df_sample_top = df_sample[df_sample['track_genre'].isin(top5_genres)]
fig_scatter = px.scatter(
    df_sample_top, x='danceability', y='popularity',
    color='track_genre', opacity=0.6,
    title="Dansabilite vs Popularite (Top 5 genres)",
    labels={'danceability': 'Dansabilite', 'popularity': 'Popularite',
            'track_genre': 'Genre'}
)
fig_scatter.update_layout(
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white'
)

#eda

# Graphique 3 — Matrice de correlation
colonnes = ['popularity', 'danceability', 'energy', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'tempo', 'duration_min',
            'dance_energy_score']
matrice_pearson = df_feat[colonnes].corr(method='pearson').round(2)
fig_corr = go.Figure(data=go.Heatmap(
    z=matrice_pearson.values,
    x=matrice_pearson.columns.tolist(),
    y=matrice_pearson.columns.tolist(),
    colorscale='RdBu', zmid=0,
    text=matrice_pearson.values.round(2),
    texttemplate='%{text}', textfont={"size": 8}
))
fig_corr.update_layout(
    title="Correlations - Dataset Spotify",
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white', height=600
)

# Graphique 4 — Distribution popularite
fig_distrib = px.histogram(
    df_feat, x='popularity', nbins=30,
    title="Distribution de la Popularite",
    color_discrete_sequence=["#20D200"]
)
fig_distrib.update_layout(
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white'
)

# Graphique 5 — Boxplot top 10 genres
top10 = df_feat.groupby('track_genre')['popularity'].mean()\
               .sort_values(ascending=False).head(10).index
df_top10 = df_feat[df_feat['track_genre'].isin(top10)]
fig_boxplot = px.box(
    df_top10, x='track_genre', y='popularity',
    color='track_genre',
    title="Distribution de la popularite par genre (Top 10)"
)
fig_boxplot.update_layout(
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white', showlegend=False,
    xaxis_tickangle=45
)

# modelisation
features = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence',
            'tempo', 'duration_min', 'dance_energy_score', 'explicit_int']
target = 'popularity'
split = int(len(df_feat) * 0.8)
X_train = df_feat[features].iloc[:split]
X_test  = df_feat[features].iloc[split:]
y_train = df_feat[target].iloc[:split]
y_test  = df_feat[target].iloc[split:]
rf_model = joblib.load('data/processed/rf_model.pkl')
y_pred = rf_model.predict(X_test)

# Graphique 6 — Importance des variables
importances = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=True)
fig_importance = px.bar(
    importances, x='importance', y='feature',
    orientation='h', color='importance',
    color_continuous_scale='Greens',
    title="Importance des variables - RandomForest"
)
fig_importance.update_layout(
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white', showlegend=False
)

# Graphique 7 — Reel vs Predit
fig_reel = go.Figure()
fig_reel.add_trace(go.Scatter(
    x=y_test, y=y_pred,
    mode='markers',
    marker=dict(color='#1DB954', opacity=0.3, size=4),
    name='Predictions'
))
fig_reel.add_trace(go.Scatter(
    x=[0, 100], y=[0, 100],
    mode='lines',
    line=dict(color='red', dash='dash'),
    name='Prediction parfaite'
))
fig_reel.update_layout(
    title="Reel vs Predit - RandomForest",
    xaxis_title="Popularite reelle",
    yaxis_title="Popularite predite",
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white'
)

# Graphique 8 — Distribution des erreurs
erreurs = y_test - y_pred
fig_erreurs = px.histogram(
    x=erreurs, nbins=40,
    title="Distribution des erreurs de prediction",
    labels={'x': 'Erreur (Reel - Predit)'},
    color_discrete_sequence=['#1DB954']
)
fig_erreurs.add_vline(x=0, line_dash='dash', line_color='red')
fig_erreurs.update_layout(
    plot_bgcolor='#111111', paper_bgcolor='#1a1a1a',
    font_color='white'
)

# METRIQUES
mae  = round(mean_absolute_error(y_test, y_pred), 3)
rmse = round(mean_squared_error(y_test, y_pred) ** 0.5, 3)
r2   = round(r2_score(y_test, y_pred), 3)

# DASHBOARD
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

sidebar = html.Div([
    html.H2("Spotify", style={'color': "#4AD95D"}),
    html.H5("Analytics Dashboard", style={'color': 'white'}),
    html.Hr(style={'borderColor': '#1DB954'}),
    html.P("NAVIGATION", style={'color': '#aaaaaa', 'fontSize': '12px'}),
    html.Div([
        html.A("Vue Generale",        href="#vue-generale",
               style={'color': 'white', 'display': 'block', 'padding': '8px',
                      'textDecoration': 'none', 'marginBottom': '5px'}),
        html.A("Visualisation",       href="#visualisation",
               style={'color': 'white', 'display': 'block', 'padding': '8px',
                      'textDecoration': 'none', 'marginBottom': '5px'}),
        html.A("EDA",                 href="#eda",
               style={'color': 'white', 'display': 'block', 'padding': '8px',
                      'textDecoration': 'none', 'marginBottom': '5px'}),
        html.A("Modelisation",        href="#modelisation",
               style={'color': 'white', 'display': 'block', 'padding': '8px',
                      'textDecoration': 'none', 'marginBottom': '5px'}),
    ]),
    html.Hr(style={'borderColor': '#333'}),
    html.P("PROJET", style={'color': '#aaaaaa', 'fontSize': '12px'}),
    html.P("97 270 chansons - 114 genres",
           style={'color': '#1DB954', 'fontSize': '11px'}),
    html.P("Danielle - Jacqueline - Aya",
           style={'color': '#aaaaaa', 'fontSize': '11px'}),
], style={
    'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0,
    'width': '220px', 'padding': '20px',
    'backgroundColor': '#181818', 'overflowY': 'auto'
})

content = html.Div([

    # METRIQUES
    html.H3("Vue Generale", id='vue-generale',
            style={'color': "#E8FFE7"}),
    dbc.Row([
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3("97 270", style={'color': "#FFFFFF", 'margin': '0'}),
            html.P("Chansons", style={'color': 'white', 'margin': '0'})
        ])], color='dark', outline=True,
           style={'borderColor': "#FFFFFF", 'textAlign': 'center'})),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3("114", style={'color': "#FFFFFF", 'margin': '0'}),
            html.P("Genres", style={'color': 'white', 'margin': '0'})
        ])], color='dark', outline=True,
           style={'borderColor': "#FFFFFF", 'textAlign': 'center'})),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3(str(r2), style={'color': "#FFFFFF", 'margin': '0'}),
            html.P("R2 Score", style={'color': 'white', 'margin': '0'})
        ])], color='dark', outline=True,
           style={'borderColor': "#FFFFFF", 'textAlign': 'center'})),
        dbc.Col(dbc.Card([dbc.CardBody([
            html.H3(str(mae), style={'color': "#FFFFFF", 'margin': '0'}),
            html.P("MAE", style={'color': 'white', 'margin': '0'})
        ])], color='dark', outline=True,
           style={'borderColor': "#FFFFFF", 'textAlign': 'center'})),
    ], style={'marginBottom': '30px'}),

    # VISUALISATION
    html.H3("Visualisation", id='visualisation',
            style={'color': "#E8FFE7", 'marginTop': '30px'}),
    dcc.Graph(figure=fig_top15),
    dcc.Graph(figure=fig_scatter),
    html.Hr(style={'borderColor': '#333'}),

    # EDA
    html.H3("EDA", id='eda',
            style={'color': "#E8FFE7", 'marginTop': '30px'}),
    dcc.Graph(figure=fig_corr),
    dcc.Graph(figure=fig_distrib),
    dcc.Graph(figure=fig_boxplot),
    html.Hr(style={'borderColor': '#333'}),

    # MODELISATION
    html.H3("Modelisation", id='modelisation',
            style={'color': "#E8FFE7", 'marginTop': '30px'}),
    dcc.Graph(figure=fig_importance),
    dcc.Graph(figure=fig_reel),
    dcc.Graph(figure=fig_erreurs),

], style={
    'marginLeft': '240px', 'padding': '25px',
    'backgroundColor': '#111111', 'minHeight': '100vh'
})

app.layout = html.Div([sidebar, content],
                      style={'backgroundColor': '#111111'})

if __name__ == '__main__':
    app.run(debug=True)