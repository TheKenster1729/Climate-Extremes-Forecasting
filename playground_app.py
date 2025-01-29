import dash
from flask import Flask
import dash_bootstrap_components as dbc
from dash import html, dcc, _dash_renderer
from analysis import RiskAssessment
import dash_mantine_components as dmc

_dash_renderer._set_react_version("18.2.0")

server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets = [dbc.themes.BOOTSTRAP])

app.layout = dmc.MantineProvider([RiskAssessment(dataset = "ERA5", var = "T2MMAX", state = "MA").risk_assessment_div_element()])

if __name__ == "__main__":
    app.run_server(debug = True)