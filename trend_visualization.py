import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from analysis import Vulcanalyzation

df = Vulcanalyzation().build_df()

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id = "month-dropdown",
        options = [
            {"label": "January", "value": "Jan"},
            {"label": "February", "value": "Feb"},
            {"label": "March", "value": "Mar"},
            {"label": "April", "value": "Apr"},
            {"label": "May", "value": "May"},
            {"label": "June", "value": "Jun"},
            {"label": "July", "value": "Jul"},
            {"label": "August", "value": "Aug"},
            {"label": "September", "value": "Sep"},
            {"label": "October", "value": "Oct"},
            {"label": "November", "value": "Nov"},
            {"label": "December", "value": "Dec"}
        ],
        value = "Jan"
    ),
    dcc.Graph(id = "trend-graph")
])

@app.callback(
    Output("trend-graph", "figure"),
    Input("month-dropdown", "value")
)
def update_graph(selected_month):
    month_df = df[df["month"] == selected_month]
    month_df["trend"] = (month_df["p_value"] < 0.05).astype(int)
    month_df["color"] = month_df["trend"].apply(lambda x: "Significant Trend" if x == 1 else "NS")

    # create a choropleth map of the trends
    fig = px.choropleth(
        month_df,
        locations = "region",
        locationmode = "USA-states",
        color = "color",
        title = f"Trends in {selected_month}",
        scope = "usa",
        color_discrete_map = {
            "Significant Trend": "#6ae66a",
            "NS": "#a83277"
        }
    )
    fig.update_layout(width = 1000, height = 600)

    return fig

if __name__ == "__main__":
    app.run_server(debug = True)