import dash
from dash import html, _dash_renderer, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask import Flask, request
import pandas as pd
import plotly.graph_objects as go
from analysis import RiskAssessment, AppFunctionsforPooledData
from styling import Naming
import dash_mantine_components as dmc
import json
import os
from pathlib import Path
_dash_renderer._set_react_version("18.2.0")

server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets = [dbc.themes.MINTY])
drawn_shapes = []
naming_df = pd.read_csv(r"region_names.csv")
states = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")["Region"].unique()
states_dict = {i: i for i in states}

# Utility function to check static plot availability
def check_static_plots_availability():
    """Check which static plots are available"""
    base_dir = Path("webapp_plots")
    if not base_dir.exists():
        return {"available": False, "message": "No static plots directory found. Run generate_static_plots.py first."}
    
    metadata_file = base_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        return {
            "available": True, 
            "message": f"Static plots available: {metadata['total_plots']} plots generated at {metadata['generated_at']}"
        }
    else:
        return {"available": False, "message": "Static plots directory exists but no metadata found."}

# Print static plot status on startup
static_status = check_static_plots_availability()
print(f"📊 Static Plot Status: {static_status['message']}")

card_built_in_regions = html.Div(id = "card-built-in-regions", children = [
                            dbc.Row(children = [
                                dbc.Col(
                                    children = [
                                    dbc.Row(children = [
                                        html.Div(id = "built-in-regions-div", children = [
                                            html.P(id = "built-in-regions-label", children = "Available Regions"),
                                            dcc.Dropdown(id = "built-in-regions",
                                                         options = [{"label": states_dict[i], "value": i} for i in states],
                                                         value = "MA",
                                                         style = {"width": "69.33%"}),
                                        ]),
                                    ]),
                                ]
                                )
                            ]
                        )
                    ]
                    )

overview_tab = html.Div(
    children = [
        html.H4("Overview"),
        html.P("This app allows you to explore and analyze regional temperature data using historical temperature data from the MERRA-2 dataset."),
        html.P("""To get started, click on the 'Existing Region' tab and select a region from the dropdown menu. You can select a scenario (Accelerated Actions, Current Trends, or Difference from CT) 
               to see the temperature trends. The Accelerated Actions scenario assumes decisive steps are taken to limit warming to 1.5° C by the end of the 21st century
               with 50% probability. Current Trends assumes nations meet their Paris Agreement targets through 2030, which is enough to slow but not halt continued growth in greenhouse gas emissions.
               The Difference from CT scenario shows the difference in temperature between the Accelerated Actions and Current Trends scenarios (i.e. the benefit of accelerated mitigation action)."""),
        html.P("""The app uses linear regression of global warming data against regional temperature data to estimate regional daily maximum temperature change. 
               Extrapolation to 2050 is accomplished by using projections of global average temperature from the Current Trends and Accelerated Actions scenarios,
               as determined by the MIT Earth Systems Model (MESM)."""),
        html.P("""The app is designed to be used by researchers, policymakers, and the general public 
               to understand the temperature trends and patterns in a region."""),
        dbc.Accordion(start_collapsed = True,
            children = [
                dbc.AccordionItem(title=  "Technical Details",
                    children = [
                        html.P("""MERRA-2 is a reanalysis dataset provided by NASA that uses satellite observations coupled with an underlying forecast model to provide
                               detailed records of Earth's climate from 1980 to the present. The MERRA-2 daily maximum temperature product ('T2MMAX') used for this app consists
                               of daily maximum temperature data observations from 1980 to 2022, gridded to a 0.5x0.625 degree latitude-longitude resolution. Historical 
                               global mean temperature used to train the regression model is obtained by taking an area-weighted average of the daily mean temperature product
                               ('T2MMEAN') over each year."""),
                        html.P("""The slope and constant coefficient of the model are obtained, as well as their standard errors, after the data are centered to 0 and scaled
                               to unit variance. Each region has its own model fit to its specific data. The model is then used to project temperature in each region under the
                               three scenarios (Accelerated Actions, Current Trends, and Difference from CT) by using the global mean temperature projections from MESM."""),
                        html.P("""There are two sources of uncertainty used in the construction of the uncertainty bands seen in the plots. The first source is from the MESM 
                               projections. MESM provides a probabilistic ensemble of 400 time series of global mean temperature through 2150 (we only use projects through 2050). 
                               The second source of uncertainty is from the regression. We sample 400 times from the normal distributions the of slope and coefficient and 
                               calculate the temperature for each region under each sample. The uncertainty bands on the graphs are the 5th and 95th percentiles of these
                               Monte Carlo simulations."""),                    
                    ]
                ),
                dbc.AccordionItem(title = "Example",
                    children = [
                        html.P("""The app currently supports the major watersheds of the United States, states of the United States, and countries/country-adjacent regions.
                               The criteria for inclusion of any of these regions is that it contains at least one point in the MERRA-2 dataset. For this example, we will look at the 
                               US state of Massachusetts."""),
                        html.P("""First, navigate to the Existing Region tab. Select the desired scenario - we will use Current Trends. Select the region for analysis
                               (it is usually easier to type in the region name rather than to scroll through the list). State names are provided as two-letter abbreviations, so
                               enter 'MA' to search for Massachusetts, and select the MA option in the dropdown menu. Press the 'Run Analysis' button to see the results."""),
                        html.Img(src = app.get_asset_url("regression_results.png"), style = {"width": "50%"}),
                        html.Img(src = app.get_asset_url("regression_projections.png"), style = {"width": "50%"}),
                        html.Br(),
                        html.P("""The image on the left shows the results of the regression analysis. The x-axes, which are shared, shows the global mean temperature for each
                               year from 1980-2022, and the MESM projections from 2023-2050. The y-axes show the local temperature of the region of interest over that same time period. Each plot shows results for one month.
                               For example, the 'Jan' plot shows how the global mean temperature for each year (x-axis) correlates to the average daily maximum temperature 
                               averaged over January only (y-axis) for that year. The x-axis label for each plot is the p-value for the slope coefficient."""),
                        html.P("""The image on the right is the more practical image - it displays the model's prediction for daily regional temperature (in this case, for
                               Massachusetts) up to 2050. We can see from the plots that the daily maximum temperature is projected to increase in each month, although the
                               increase is almost nothing for May (and checking the plot on the left, the regression slope is not statistically significant for May, so the model
                               is inconclusive vis a vis temperature increase in May for Massachusetts)."""),
                        html.P("""In both plots, the orange line shows the median prediction and the shaded uncertainty bands show the 5th/95th percentiles."""),
                    ]
                ),
                dbc.AccordionItem(title = "Attributions",
                    children = [
                        html.P("""This app was designed and coded by Kenneth Cox at the MIT Center for Sustainability Science and Strategy (CS3) under the supervision of 
                               Jennifer Morris, Adam Schlosser, and Xiang Gao. MESM temperature projections were provided by Popat Salunke, also at the Center. Correspondence
                               should be directed to Kenneth Cox at kcox1729@mit.edu.""")
                    ]
                )
            ]
        )
    ]
)

app.layout = dmc.MantineProvider(html.Div(
    children = [
        html.H4("Daily Max/Mean/Min Temperature Forecasting", className = "bg-primary text-white p-2 mb-2 text-center"),
        html.Br(),
        dbc.Tabs(
            children = [
                dbc.Tab(label = "Overview", children = [overview_tab], style = {"fontSize": "1.2em", "margin": "20px"}),
                dbc.Tab(label = "Visualize State Trends", 
                        children = [
                            html.Div(style = {"margin": "20px"},
                                children = [
                                    dbc.Row(
                                        children = [
                                            dbc.Col(
                                                children = [
                                                    dbc.Card(style = {"marginLeft": "20px"},
                                                        children = [
                                                            dbc.CardBody(style = {"marginLeft": "20px", "marginRight": "20px"},
                                                                children = [
                                                                    dbc.Row(
                                                                        children = [
                                                                            dbc.Col(
                                                                                children = [
                                                                                    dbc.Row(
                                                                                        children = [
                                                                                            dbc.Col(
                                                                                                children = [
                                                                                                    html.P("Scenario Selection", className = "primary"),
                                                                                                    dcc.Dropdown(id = "scenarios-dropdown-built-in", options = [{"label": "Accelerated Actions", "value": "aa"}, {"label": "Current Trends", "value": "ct"}, {"label": "Difference From CT", "value": "diff"}], value = "ct",
                                                                                                        style = {"width": "100%"}),
                                                                                            ]
                                                                                        ),
                                                                                        dbc.Col(
                                                                                            children = [
                                                                                                html.P("Variable Selection", className = "primary"),
                                                                                                dcc.Dropdown(id = "variable-dropdown-built-in", options = [{"label": "Daily Max", "value": "T2MMAX"}, {"label": "Daily Mean", "value": "T2MMEAN"}, {"label": "Daily Min", "value": "T2MMIN"}], value = "T2MMAX",
                                                                                                    style = {"width": "100%"}),
                                                                                            ]
                                                                                        )
                                                                                        ]
                                                                                    ),
                                                                                    html.Br(),
                                                                                    html.Div(id = "built-in-region-menus", children = [card_built_in_regions]),
                                                                                    html.Br(),
                                                                                    dbc.Row(
                                                                                        children = [
                                                                                            dbc.Col(
                                                                                                children = [
                                                                                                    dcc.Store(id = "region-name-store", storage_type = "session", data = 0)
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
                                        dbc.Col(id = "risk-assessment-area",
                                            children = [
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
                                                            dbc.Col(
                                                                children = [
                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-temp-div-built-in", children = [html.Iframe(id = "analysis-graph-temp-built-in", width = "100%", height = "600")], hidden = True)], size = "sm")
                                                                ]
                                                            ),
                                                            dbc.Col(
                                                                children = [
                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-year-div-built-in", children = [html.Iframe(id = "analysis-graph-year-built-in", width = "100%", height = "600")], hidden = True)], size = "sm")
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
                #dbc.Tab(label = "Visualize State Data", children = [state_tab])
                ]
            )
        ]
    )
)

# callback for built-in regions - now serves pre-generated static HTML
@app.callback(Output("analysis-graph-temp-built-in", "srcDoc"),
              Output("analysis-graph-year-built-in", "srcDoc"),
              Output("analysis-graph-temp-div-built-in", "hidden"),
              Output("analysis-graph-year-div-built-in", "hidden"),
              Input("variable-dropdown-built-in", "value"),
              Input("built-in-regions", "value"),
              Input("scenarios-dropdown-built-in", "value"))
def update_analysis_graph(var, region_name, scenario):
    # For now, only supporting T2MMAX variable since that's what we pre-generated
    if var != "T2MMAX" or not region_name or not scenario:
        return "", "", True, True
    
    # Get paths to pre-generated HTML files
    base_dir = "webapp_plots"
    temp_file_path = Path(base_dir) / scenario / f"{region_name}_temp.html"
    year_file_path = Path(base_dir) / scenario / f"{region_name}_year.html"
    
    # Read the HTML files
    temp_html = ""
    year_html = ""
    
    try:
        if temp_file_path.exists():
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                temp_html = f.read()
        else:
            print(f"⚠️  Static file not found: {temp_file_path}")
        
        if year_file_path.exists():
            with open(year_file_path, 'r', encoding='utf-8') as f:
                year_html = f.read()
        else:
            print(f"⚠️  Static file not found: {year_file_path}")
                
        # If we have at least one plot, show the component
        if temp_html or year_html:
            return temp_html, year_html, False, False
        else:
            print(f"⚠️  No static plots found for {region_name}/{scenario}, falling back to dynamic generation")
            
    except Exception as e:
        print(f"❌ Error loading static plots for {region_name}/{scenario}: {e}")
    
    # Fallback to dynamic generation if static files don't exist
    try:
        print(f"🔄 Generating plots dynamically for {region_name}/{scenario}")
        by_temp, by_year = AppFunctionsforPooledData(var = var, scenario = scenario).make_plots(region_name)
        return by_temp.to_html(include_plotlyjs='cdn'), by_year.to_html(include_plotlyjs='cdn'), False, False
    except Exception as e:
        print(f"❌ Dynamic generation failed: {e}")
        return "", "", True, True

# callback for risk assessment
@app.callback(Output("risk-assessment-area", "children"),
              Output("region-name-store", "data"),
              Input("built-in-regions", "value"),
              State("region-name-store", "data"))
def update_risk_assessment(region_name, region_name_store):
    region_name_store += 1
    return RiskAssessment(dataset = "MERRA2", var = "T2MMAX", state = region_name.upper()).risk_assessment_div_element(region_name_store), region_name_store

if __name__ == "__main__":
    app.run_server(debug = True)