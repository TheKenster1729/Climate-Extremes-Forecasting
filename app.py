import dash
from dash import html, _dash_renderer, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from flask import Flask, request
import pandas as pd
import plotly.graph_objects as go
import dash_mantine_components as dmc
import json
import os
from pathlib import Path
from dash import html, dcc, _dash_renderer
from dash_mantine_components import Text
_dash_renderer._set_react_version("18.2.0")

server = Flask(__name__)
app = dash.Dash(__name__, server = server, external_stylesheets = [dbc.themes.MINTY])
risk_assessment_df = pd.read_csv(r"risk_assessment.csv")
state_populations = pd.read_csv(r"state_populations.csv")
state_flowers = pd.read_csv(r"state_flowers.csv")
state_names_df = pd.read_csv(r"state_cmi.csv")
state_names_df = state_names_df[state_names_df["state"] != "D.C."]
abbreviation_dict = {key: value for value, key in zip(state_names_df["state"], state_names_df["abbreviation"])}
states = {value: key for key, value in abbreviation_dict.items()}

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
print(f"Static Plot Status: {static_status['message']}")

card_built_in_regions = html.Div(id = "card-built-in-regions", children = [
                            dbc.Row(children = [
                                dbc.Col(
                                    children = [
                                    dbc.Row(children = [
                                        html.Div(id = "built-in-regions-div", children = [
                                            html.P(id = "built-in-regions-label", children = "Available Regions"),
                                            dcc.Dropdown(id = "built-in-regions",
                                                         options = [{"label": i, "value": states[i]} for i in states],
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
        html.P("This app allows you to explore and analyze state-level temperature data using historical temperature data from the MERRA-2 and ERA5 reanalysis datasets."),
        html.P("""To get started, click on the 'Visualize State Trends' tab and select a state from the dropdown menu. You can select a scenario (Accelerated Actions or Current Trends) 
               to see the temperature trends. The Accelerated Actions scenario assumes decisive steps are taken to limit warming to 1.5° C by the end of the 21st century
               with 50% probability. Current Trends assumes nations meet their Paris Agreement targets through 2030, which is enough to slow but not halt continued growth in greenhouse gas emissions."""),
        html.P("""The app uses linear regression of global warming data against state-level temperature data to estimate regional temperature change.
               Extrapolation to 2050 is accomplished by using projections of global average temperature from the Current Trends and Accelerated Actions scenarios,
               as determined by the MIT Earth Systems Model (MESM)."""),
        html.P("""The app is designed to be used by researchers, policymakers, and the general public 
               to understand the temperature trends and patterns in a region."""),
        dbc.Accordion(start_collapsed = True,
            children = [
                dbc.AccordionItem(title=  "Technical Details",
                    children = [
                        html.P("""MERRA-2 and ERA5 are reanalysis datasets, i.e., they use satellite observations coupled with an underlying forecast model to provide
                               detailed historical records of Earth's climate. At a high level, we extracted monthly minimum, mean, and maximum temperature data from 1980-2022
                               for each state in the United States from both the MERRA-2 and ERA5 datasets, along with the global mean temperature over the same period for each dataset.
                               We then regressed the state-level temperature data against the global mean temperature for each month, and used the slope and constant coefficient of the model
                               to project temperature in each state under the two MESM scenarios by using the global mean temperature projections from MESM."""),
                        html.P("""A more detailed breakdown of the methodology is given in graphical form below. It is easiest to understand the methodology by focusing on just one state and one month
                               (Missouri in January is used as an example). We take the monthly minimum, mean, and maximum temperatures for Missouri in January over 1980-2022 and regress them individually against global
                               mean temperature to get a trend and an intercept for both the MERRA-2 and ERA5 datasets. We use a bootstrap approach (doing 500 total regressions) to estimate
                               the uncertainty of the trend and intercept. Then, we take the weighted average of the two trends, where the weights are the inverse variances of the trends. (That way,
                               we give more weight to the trend with the lower uncertainty). We do the same for the intercepts; this produces a single linear model that predicts the average January
                               temperature for Missouri. By taking the aggregate of the two reanalysis datasets, our approach is designed to find the true underlying trend and intercept, rather than
                               relying on the performance of a single dataset."""),
                        html.P("""We repeat this process for each state and each month. Because the trend in regional temperature change, as opposed to the intercept, is more valuable to policymakers,
                               our analysis in the companion paper focuses on the trend. The paper singles out maximum temperatures, but our process also allows us to show the trends for each
                               state and month for minimum and mean temperatures, as the graphic below demonstrates (using minimum temperature as an example). Finally, we use the trend and intercept 
                               for each state and month to project the temperature in each state under the two MESM scenarios by using the global mean temperature projections from MESM."""),
                        html.Br(),
                        html.Img(src = app.get_asset_url("updated_methodology_figure.svg"), style = {"width": "100%"}),
                    ]
                ),
                dbc.AccordionItem(title = "Features",
                    children = [
                        html.P("""The plots are produced with the Plotly graphing library, which provides several interactive features. The legend, at the top right of the plots, is interactive:
                               by selecting and unselecting the legend items, you can toggle the visibility of the data points and uncertainty bands in the plot. \"PI\" refers to the prediction interval,
                               which captures the uncertainty in predicting a particular output value; we show the prediction interval corresponding to the middle 95% of predicted output values.
                               The confidence interval (CI) captures the uncertainty in the mean response, so the uncertainty bands are narrower."""),
                        html.P("""The other useful feature of the plot is the ability to download plots as PNGs. If you hover over the plots, at the top you will see a row of options to the right. One
                               of them looks like a camera - clicking on it will download the current plot as a PNG. """),
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
        html.H4("State Temperature Trends Visualizer", className = "bg-primary text-white p-2 mb-2 text-center"),
        html.Br(),
        html.H5("Produced by the MIT Center for Sustainability Science and Strategy", className = "text-left", style = {"marginLeft": "20px"}),
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
                                                                                                    dcc.Dropdown(id = "scenarios-dropdown-built-in", options = [{"label": "Accelerated Actions", "value": "aa"}, {"label": "Current Trends", "value": "ct"}], value = "ct",
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
                                                html.P("Select a state to begin")
                                                ]
                                            )
                                        ]
                                    ),
                                    dbc.Row(
                                        children = [
                                            dbc.Col(
                                                children = [
                                                    # Temperature plot (by-temp) on top
                                                    dbc.Row(
                                                        children = [
                                                            dbc.Col(
                                                                children = [
                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-temp-div-built-in", children = [html.Iframe(id = "analysis-graph-temp-built-in", width = "100%", height = "800")], hidden = True)], size = "sm")
                                                                ]
                                                            )
                                                        ]
                                                    ),
                                                    # Year plot (by-year) underneath
                                                    dbc.Row(
                                                        children = [
                                                            dbc.Col(
                                                                children = [
                                                                    dbc.Spinner(children = [html.Div(id = "analysis-graph-year-div-built-in", children = [html.Iframe(id = "analysis-graph-year-built-in", width = "100%", height = "800")], hidden = True)], size = "sm")
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
    # Check if all required inputs are provided
    if not var or not region_name or not scenario:
        return "", "", True, True
    
    # Get paths to pre-generated HTML files using the new directory structure
    base_dir = "webapp_plots"
    var_folders = {"T2MMAX": "Max", "T2MMEAN": "Mean", "T2MMIN": "Min"}
    var_folder = var_folders.get(var, "Max")
    temp_file_path = Path(base_dir) / scenario / var_folder / f"{region_name}_temp.html"
    year_file_path = Path(base_dir) / scenario / var_folder / f"{region_name}_year.html"
    
    # Read the HTML files
    temp_html = ""
    year_html = ""

    try:
        if temp_file_path.exists():
            with open(temp_file_path, 'r', encoding='utf-8') as f:
                temp_html = f.read()
        else:
            print(f"Static file not found: {temp_file_path}")
        
        if year_file_path.exists():
            with open(year_file_path, 'r', encoding='utf-8') as f:
                year_html = f.read()
        else:
            print(f"Static file not found: {year_file_path}")
                
        # If we have at least one plot, show the component
        if temp_html or year_html:
            return temp_html, year_html, False, False
        else:
            print(f"No static plots found for {region_name}/{scenario}, falling back to dynamic generation")
            
    except Exception as e:
        print(f"Error loading static plots for {region_name}/{scenario}: {e}")
    
        # Fallback to dynamic generation if static files don't exist
        try:
            print(f"Generating plots dynamically for {region_name}/{scenario}/{var}")
            by_temp, by_year = AppFunctionsforPooledData(var = var, scenario = scenario).make_plots(region_name)
            return by_temp.to_html(include_plotlyjs='cdn'), by_year.to_html(include_plotlyjs='cdn'), False, False
        except Exception as e:
            print(f"Dynamic generation failed for {var}: {e}")
            return "", "", True, True

# callback for risk assessment
# use risk assessment csv
@app.callback(Output("risk-assessment-area", "children"),
              Output("region-name-store", "data"),
              Input("built-in-regions", "value"),
              State("region-name-store", "data"),
              prevent_initial_call = True)
def update_risk_assessment(region_name, region_name_store):
    region_name_store += 1
    full_region_name = abbreviation_dict[region_name]
    risk = risk_assessment_df[risk_assessment_df["Region"] == region_name]["risk"].values[0]
    color = risk_assessment_df[risk_assessment_df["Region"] == region_name]["risk_color"].values[0]
    div_element = html.Div(
        children = [Text(f"{full_region_name}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 30, "color": "black"}, id = f"state-name-{region_name_store}"),
                    Text(f"Population: {state_populations[state_populations['State'] == full_region_name]['Population'].values[0]} ｜ State flower: {state_flowers[state_flowers['State'] == full_region_name]['Common name'].values[0]}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 20, "color": "black"}, id = f"state-info-{region_name_store}"),
                    html.Div(children = [Text(f"Warming Risk:", style = {"fontSize": 20, "color": "black"}, className = "animate__animated animate__fadeInRightBig animate__slow", id = f"risk-label-{region_name_store}"), Text(f"{risk}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 20, "color": color, "marginLeft": "10px"}, id = f"risk-value-{region_name_store}")], style = {"display": "flex", "alignItems": "left"})]
    )
    return div_element, region_name_store


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050) # for the webapp