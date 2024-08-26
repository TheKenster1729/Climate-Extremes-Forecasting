import dash
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash import dcc
from flask import Flask, request
import pandas as pd
from data_retrieval import CollectRegionalData
from graphing_utils import TimeSeriesFromAppData
import plotly.graph_objects as go
from analysis import RegressionModel
from styling import Naming

print("HELLO WORLD")

server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets = [dbc.themes.MINTY])
drawn_shapes = []
naming_df = pd.read_csv(r"region_names.csv")

card_custom_regions = html.Div(id = "card-custom-regions", children = [
                            dbc.Row(children = [
                                dbc.Col(width = 6,
                                    children = [
                                    dbc.Row(children = [
                                       html.Iframe(id = "map", srcDoc = open("map.html", "r").read(), height = 600)
                                    ]),
                                    html.Br(),
                                    html.Div(id = "region-name-div", children = [
                                        dbc.Row(children = [
                                            dbc.Col(width = 6,
                                            children = [
                                                dbc.Input(id = "region-name-input", placeholder = "Enter Region Name", type = "text", style = {"width": "100%"}),
                                            ]),
                                        dbc.Col(width = 4,
                                            children = [
                                                dbc.Button(id = "region-name-button", children = "Name Region", color = "primary", n_clicks = 0),
                                            ])
                                        ])
                                    ])
                                ]
                            )
                        ]
                    )
                        ]
                    )

card_built_in_regions = html.Div(id = "card-built-in-regions", children = [
                            dbc.Row(children = [
                                dbc.Col(
                                    children = [
                                    dbc.Row(children = [
                                        html.Div(id = "built-in-regions-div", children = [
                                            html.P(id = "built-in-regions-label", children = "Available Regions"),
                                            dcc.Dropdown(id = "built-in-regions", options = [{"label": i[2], "value": i[1]} for i in naming_df.values], value = "newenglandregion", style = {"width": "60%"}),
                                        ]),
                                    ]),
                                ]
                                )
                            ]
                        )
                    ]
                    )

app.layout = html.Div(
    children = [
        html.H4("Daily Max Temperature Forecasting", className = "bg-primary text-white p-2 mb-2 text-center"),
        html.Br(),
        dbc.Tabs(
            children = [
                dbc.Tab(label = "Overview"),
                dbc.Tab(label = "Custom Region", 
                    children = [
                        html.Div(
                                children = [
                                    dbc.Card(style = {"margin-left": "20px", "margin-right": "20px"},
                                        children = [
                                            dbc.CardBody(style = {"margin-left": "20px", "margin-right": "20px"},
                                                children = [
                                                    dbc.Row(
                                                        children = [
                                                            dbc.Col(
                                                                children = [
                                                                    dbc.Row(
                                                                        children = [
                                                                            html.P("Scenario Selection", className = "primary"),
                                                                            dcc.Dropdown(id = "scenarios-dropdown-custom", options = [{"label": "Accelerated Actions", "value": "aa"}, {"label": "Current Trends", "value": "ct"}
                                                                                                                                      , {"label": "Difference From CT", "value": "diff"}], value = "ct",
                                                                                style = {"width": "60%"}),
                                                                            ]
                                                                        ),
                                                            html.Br(),
                                                            html.Div(id = "custom-region-menus", children = [card_custom_regions]),
                                                            html.Br(),
                                                            dbc.Row(
                                                                children = [
                                                                    dbc.Col(
                                                                        children = [
                                                                            dbc.Button(id = "run-analysis-button-custom", children = ["Run Analysis"], color = "primary", n_clicks = 0),
                                                                        ]
                                                                    )
                                                                ]
                                                            )
                                                        ]
                                                    ),
                                                    html.Br(),
                                                    dbc.Row(
                                                        children = [
                                                            dbc.Col(
                                                                children = [
                                                                    html.H3(id = "region-name-custom", style = {"marginTop": "20px"}),
                                                                    dbc.Row(
                                                                        children = [
                                                                            dbc.Col(
                                                                                children = [
                                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-temp-div-custom", children = [dcc.Graph(id = "analysis-graph-temp-custom")], hidden = True)], size = "sm")
                                                                                ]
                                                                            ),
                                                                            dbc.Col(
                                                                                children = [
                                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-year-div-custom", children = [dcc.Graph(id = "analysis-graph-year-custom")], hidden = True)], size = "sm")
                                                                                ]
                                                                            )
                                                                        ]
                                                                    )
                                                                ]
                                                            )
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            dbc.Tab(label = "Existing Region", 
                    children = [
                        html.Div(
                            children = [
                                dbc.Card(style = {"margin-left": "20px", "margin-right": "20px"},
                                    children = [
                                        dbc.CardBody(style = {"margin-left": "20px", "margin-right": "20px"},
                                            children = [
                                                dbc.Row(
                                                    children = [
                                                        dbc.Col(
                                                            children = [
                                                                dbc.Row(
                                                                    children = [
                                                                        html.P("Scenario Selection", className = "primary"),
                                                                        dcc.Dropdown(id = "scenarios-dropdown-built-in", options = [{"label": "Accelerated Actions", "value": "aa"}, {"label": "Current Trends", "value": "ct"}, {"label": "Difference From CT", "value": "diff"}], value = "ct",
                                                                            style = {"width": "60%"}),
                                                                        html.Br(),
                                                                    ]
                                                                ),
                                                                html.Br(),
                                                                html.Div(id = "built-in-region-menus", children = [card_built_in_regions]),
                                                                html.Br(),
                                                                dbc.Row(
                                                                    children = [
                                                                        dbc.Col(
                                                                            children = [
                                                                                dbc.Button(id = "run-analysis-button-built-in", children = ["Run Analysis"], color = "primary", n_clicks = 0),
                                                                        ]
                                                                    )
                                                                ]
                                                            )
                                                        ]
                                                    ),
                                                    dbc.Row(
                                                        children = [
                                                            dbc.Col(
                                                                children = [
                                                                    dbc.Row(
                                                                        children = [
                                                                            html.H3(id = "region-name-built-in", style = {"marginTop": "20px"}),
                                                                            dbc.Col(
                                                                                children = [
                                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-temp-div-built-in", children = [dcc.Graph(id = "analysis-graph-temp-built-in")], hidden = True)], size = "sm")
                                                                                ]
                                                                            ),
                                                                            dbc.Col(
                                                                                children = [
                                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-year-div-built-in", children = [dcc.Graph(id = "analysis-graph-year-built-in")], hidden = True)], size = "sm")
                                                                                ]
                                                                            )
                                                                        ]
                                                                    )
                                                                ]
                                                            )
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
        )
    ]
)

# update region name - assume that the latest region drawn is the one to be named
@app.callback(Output("region-name-input", "value"),
              State("region-name-input", "value"),
              Input("region-name-button", "n_clicks"),
              prevent_initial_call = True)
def update_region_name(region_name, n_clicks):
    if n_clicks > 0:
        drawn_shapes[-1]["name"] = region_name
    return None

@server.route('/drawn_shapes', methods=['POST'])
def handle_drawn_shapes():
    data = request.get_json()
    global drawn_shapes
    drawn_shapes.append(data)
    # Process the GeoJSON data as needed
    return "OK"

# callback for custom regions
@app.callback(Output("analysis-graph-temp-custom", "figure"),
              Output("analysis-graph-year-custom", "figure"),
              Output("analysis-graph-temp-div-custom", "hidden"),
              Output("analysis-graph-year-div-custom", "hidden"),
              Output("region-name-custom", "children"),
              Output("map", "srcDoc"),
              Input("run-analysis-button-custom", "n_clicks"),
              State("scenarios-dropdown-custom", "value"),
              prevent_initial_call = True)
def update_analysis_graph(n_clicks, scenario):
    # use last drawn shape as polygon
    region_name = drawn_shapes[-1]["name"] if len(drawn_shapes) > 0 else None

    data = CollectRegionalData(drawn_shapes[-1]["geometry"], from_app = True).aggregate_inside_points_temp_data()
    by_temp, by_year = RegressionModel(data, from_app = True, scenario = scenario, show_year_graph = True).average_monthly_max_temp_regression()

    return by_temp, by_year, False, False, "Region: " + region_name, open("map.html", "r").read()

# callback for built-in regions
@app.callback(Output("analysis-graph-temp-built-in", "figure"),
              Output("analysis-graph-year-built-in", "figure"),
              Output("analysis-graph-temp-div-built-in", "hidden"),
              Output("analysis-graph-year-div-built-in", "hidden"),
              Output("region-name-built-in", "children"),
              Input("run-analysis-button-built-in", "n_clicks"),
              State("built-in-regions", "value"),
              State("scenarios-dropdown-built-in", "value"),
              prevent_initial_call = True)
def update_analysis_graph(n_clicks, region_name, scenario):
    path_to_data = r"MERRA2/JSON Files/Regional Aggregates/{}_average_t2mmax.json".format(region_name)
    by_temp, by_year = RegressionModel(path_to_data, scenario = scenario, show_year_graph = True).main()

    return by_temp, by_year, False, False, "Region: " + naming_df[naming_df["Stem"] == region_name]["Region Name"]

if __name__ == "__main__":
    app.run_server(debug = True)