from dash import Dash, dcc, html, MATCH, ALL, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import matplotlib.ticker as mtick
from pandas_datareader import data as pdr
import investpy as inv
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url
from tabulate import tabulate


# stylesheet with the .dbc class
dbc_css = "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates/dbc.min.css"


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])

retornos_um_ano = pd.read_csv('https://raw.githubusercontent.com/gp3try/Projeto_Markowitz/main/retornos_um_ano.csv')

ativos = list(retornos_um_ano.columns.values)


app.layout = html.Div([
    html.Label('Selecione seus ativos '),
    dcc.Dropdown(ativos, multi=True, id='dropdown_ativos'),
    html.Button(id='add_stock', n_clicks=0, children='Submit'),
    html.Br(),
    dcc.Graph(id='preco_acoes'),
    html.Hr(),
    html.Div(id='portfolio_otimizado'),
    html.Br(),
    html.Button(id='gerar_markowitz', n_clicks=0, children='Otimizar Portfólio'),
    dcc.Graph(id='grafico_markowitz'),
    dcc.Store(id='carteira_usuario')
    ])

@app.callback(
    Output("carteira_usuario", "data"),
     Input('add_stock', 'n_clicks'),
     State('dropdown_ativos', 'value'))
def update_table(n_clicks, ativos):
    if n_clicks in [0, None]:
        raise PreventUpdate
    else:
        df = pd.DataFrame()
        df['Date'] = retornos_um_ano['Date']
        for i in ativos:
            df[i] = retornos_um_ano[i]
        df.set_index('Date')
        return df.to_json(date_format='iso', orient='split')

@app.callback(
    Output('preco_acoes', 'figure'),
    Input('carteira_usuario', 'data'))
def criando_grafico(df):
    dff = pd.read_json(df, orient='split')
    fig = px.line(dff, x='Date', y=dff.columns)
    return fig


@app.callback(
    Output('portfolio_otimizado', 'children'),
    Output('grafico_markowitz', 'figure'),
    Input('gerar_markowitz', 'n_clicks'),
    State('carteira_usuario', 'data'))
def gerando_graficos(n_clicks, carteira):
    dff = pd.read_json(carteira, orient='split')
    
    numero_de_carteiras = 100000
    tabela_retornos_esperados = np.zeros(numero_de_carteiras)
    tabela_volatilidades_esperadas = np.zeros(numero_de_carteiras)
    tabela_sharpe = np.zeros(numero_de_carteiras)
    tabela_pesos = np.zeros((numero_de_carteiras, dff.shape[1] -1))
    media_retornos = dff.mean()
    matriz_cov = dff.cov()

    for k in range(numero_de_carteiras):
        pesos = np.random.random(dff.shape[1] -1)
        pesos = pesos/np.sum(pesos)
        tabela_pesos[k, :] = pesos
        
        tabela_retornos_esperados[k] = np.sum(media_retornos * pesos * 252)
        tabela_volatilidades_esperadas[k] = np.sqrt(np.dot(pesos.T, np.dot(matriz_cov * 252, pesos)))
        tabela_sharpe[k] = tabela_retornos_esperados[k]/tabela_volatilidades_esperadas[k]
        sharpe_maximo = tabela_sharpe.argmax()
        tabela_pesos[sharpe_maximo]

    fig = px.scatter(x=tabela_volatilidades_esperadas, y=tabela_retornos_esperados, color=tabela_sharpe)
    return u'A alocação ótima é dada por {}'.format(tabela_pesos[sharpe_maximo]), fig


if __name__ == '__main__':
    app.run_server(debug=True)

