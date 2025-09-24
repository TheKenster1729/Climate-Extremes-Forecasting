import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import pandas as pd
import numpy as np
from scipy.stats import t
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, Tuple, Dict
from plotly.colors import n_colors, hex_to_rgb
import pymannkendall as mk
from scipy.stats import pearsonr

class AppFunctionsforPooledData:
    def __init__(self, scenario, var = "T2MMAX", end_year = 2050):
        self.scenario = scenario
        self.var = var
        self.end_year = end_year
        
        # Load appropriate data file based on variable
        if var == "T2MMAX":
            self.data = pd.read_csv(r"full_processed_data_t2mmax.csv")
            self.regression_results = pd.read_csv(r"Regression Results/pooled_bootstrap_results_t2mmax.csv")
            self.uncertainty_intervals = pd.read_csv(r"Regression Results/uncertainty_intervals_t2mmax.csv")
        elif var == "T2MMEAN":
            self.data = pd.read_csv(r"full_processed_data_t2mmean.csv")
            self.regression_results = pd.read_csv(r"Regression Results/pooled_bootstrap_results_t2mmean.csv")
            self.uncertainty_intervals = pd.read_csv(r"Regression Results/uncertainty_intervals_t2mmean_percentiles.csv")
        elif var == "T2MMIN":
            self.data = pd.read_csv(r"full_processed_data_t2mmin.csv")
            self.regression_results = pd.read_csv(r"Regression Results/pooled_bootstrap_results_t2mmin.csv")
            self.uncertainty_intervals = pd.read_csv(r"Regression Results/uncertainty_intervals_t2mmin_percentiles.csv")
        else:
            raise ValueError(f"Unsupported variable: {var}. Use T2MMAX, T2MMEAN, or T2MMIN.")
        
        self.regression_years = self.data.Year.unique()
        # global data is the same for all months, so we can use any month to get the global mean temp
        self.era5_global_mean_temp = self.data[(self.data["Year"].isin(self.regression_years)) & (self.data["Dataset"] == "era5") & (self.data["Month"] == "Jan")]["Global_Temp"].mean()
        self.merra2_global_mean_temp = self.data[(self.data["Year"].isin(self.regression_years)) & (self.data["Dataset"] == "merra2") & (self.data["Month"] == "Jan")]["Global_Temp"].mean()

    def get_merra2_historical_data(self, region):
        m2_data = self.data[(self.data["Dataset"] == "merra2") & (self.data["Region"] == region)]
        return m2_data

    def get_era5_historical_data(self, region):
        e5_data = self.data[(self.data["Dataset"] == "era5") & (self.data["Region"] == region)]
        return e5_data

    def _load_scenario_temps(self, scenario: str) -> pd.DataFrame:
        """Load global temperature projections for a scenario (aa or ct)."""
        try:
            filename = f"{scenario}_t2m.csv"
            df = pd.read_csv(filename)
            # Assume first column is Year, rest are runs
            df = df[(df.iloc[:, 0] >= 2023) & (df.iloc[:, 0] <= 2050)]
            return df
        except Exception:
            return pd.DataFrame()

    def _get_bootstrap_coeffs(self, region: str, month: str) -> Tuple[np.ndarray, np.ndarray, float]:
        """Get bootstrap coefficients and residual std for region/month."""
        qi = self.uncertainty_intervals[(self.uncertainty_intervals["Region"] == region) & 
                                       (self.uncertainty_intervals["Month"] == month)]
        if not qi.empty:
            b0_med, b1_med = qi["Intercept_50th"].iloc[0], qi["Slope_50th"].iloc[0]
            b0_lo, b0_hi = qi["Intercept_5th"].iloc[0], qi["Intercept_95th"].iloc[0]
            b1_lo, b1_hi = qi["Slope_5th"].iloc[0], qi["Slope_95th"].iloc[0]
            
            # Approximate standard deviations from 90% intervals
            z = 1.645  # 90% interval
            b0_sd = max((b0_hi - b0_lo) / (2 * z), 1e-6)
            b1_sd = max((b1_hi - b1_lo) / (2 * z), 1e-6)
            
            # Generate approximate bootstrap samples
            rng = np.random.default_rng(42)
            ints = rng.normal(b0_med, b0_sd, 1000)
            slps = rng.normal(b1_med, b1_sd, 1000)
            
            # Estimate residual std from historical data
            hist = self.data[(self.data["Region"] == region) & (self.data["Month"] == month)]
            if not hist.empty:
                mu = hist["Global_Temp"].mean()
                y_hat = b0_med + b1_med * (hist["Global_Temp"] - mu)
                res = hist["Average_Temperature"] - y_hat
                sigma = max(np.std(res, ddof=1), 1e-6)
            else:
                sigma = 1.0
                
            return ints, slps, sigma
        
        return np.array([0.0]), np.array([0.0]), 1.0

    def _add_projection_bands(self, fig, region: str, month: str, scenario: str, 
                             color: str, name: str, row: int, col: int):
        """Add 95% prediction interval bands for future projections."""
        # Load scenario temperature projections
        temp_df = self._load_scenario_temps(scenario)
        if temp_df.empty:
            return
            
        # Get bootstrap coefficients
        ints, slps, sigma = self._get_bootstrap_coeffs(region, month)
        
        # Calculate centering mean from historical data
        hist_data = self.data[(self.data["Region"] == region) & (self.data["Month"] == month)]
        center_mean = hist_data["Global_Temp"].mean() if not hist_data.empty else 0.0
        
        # For each year, calculate median global temp and 95% CI
        years = temp_df.iloc[:, 0].values
        x_medians = []
        y_lowers = []
        y_uppers = []
        
        for year in years:
            year_temps = temp_df[temp_df.iloc[:, 0] == year].iloc[:, 1:].values.flatten()
            year_temps = year_temps[~np.isnan(year_temps)]
            
            if len(year_temps) > 0:
                x_median = np.median(year_temps)
                x_medians.append(x_median)
                
                # Monte Carlo for prediction interval
                n_samples = 1000
                rng = np.random.default_rng(123)
                
                # Sample from bootstrap coefficients
                boot_idx = rng.integers(0, len(ints), n_samples)
                sampled_ints = ints[boot_idx]
                sampled_slps = slps[boot_idx]
                
                # Sample from predictor uncertainty
                temp_samples = rng.choice(year_temps, n_samples)
                
                # Calculate predictions with all uncertainties
                y_pred = sampled_ints + sampled_slps * (temp_samples - center_mean)
                y_pred += rng.normal(0, sigma, n_samples)  # Add residual noise

                # Calculate 95% prediction interval
                y_lower, y_upper = np.percentile(y_pred, [2.5, 97.5])
                y_lowers.append(y_lower)
                y_uppers.append(y_upper)

        if x_medians:
            # Add prediction band
            fig.add_trace(go.Scatter(
                x=np.concatenate([x_medians, x_medians[::-1]]),
                y=np.concatenate([y_uppers, y_lowers[::-1]]),
                fill='toself', fillcolor=color, opacity=0.3,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', showlegend=(row == 1 and col == 1), 
                name=f"{name} 95% PI",
                legendgroup=f"{scenario}_projection"
            ), row=row, col=col)
            
            # Add median projection line
            y_medians = [(ints.mean() + slps.mean() * (x - center_mean)) for x in x_medians]
            fig.add_trace(go.Scatter(
                x=x_medians, y=y_medians, mode='lines',
                line=dict(color=color.replace('rgba', 'rgb').replace(', 0.3', ''), width=2),
                name=f"{name} Median", showlegend=(row == 1 and col == 1),
                legendgroup=f"{scenario}_projection"
            ), row=row, col=col)

    def make_by_temp_plot(self, region):
        # Use PlotlySlopeMap for historical data and regression
        try:
            # Use variable-specific prediction bands file
            if self.var == "T2MMAX":
                prediction_bands_file = r"Regression Results/uncertainty_intervals_with_prediction_bands_t2mmax.csv"
            elif self.var == "T2MMEAN":
                prediction_bands_file = r"Regression Results/uncertainty_intervals_with_prediction_bands_t2mmean.csv"
            elif self.var == "T2MMIN":
                prediction_bands_file = r"Regression Results/uncertainty_intervals_with_prediction_bands_t2mmin.csv"
            else:
                # Fallback to old naming convention
                prediction_bands_file = r"Regression Results/uncertainty_intervals_with_prediction_bands.csv"
            
            coeff_df = pd.read_csv(prediction_bands_file)
            slope_map = PlotlySlopeMap(coeff_df=coeff_df, hist_df=self.data)
            fig = slope_map.create_combined_grid(state=region)
        except Exception:
            # Fallback to manual plotting if PlotlySlopeMap fails
            months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
            fig = make_subplots(rows=3, cols=4, subplot_titles=months, vertical_spacing=0.05)
            
            historical_df = self.data[self.data["Region"] == region].copy()
            for i, month in enumerate(months):
                month_data = historical_df[historical_df["Month"] == month]
                row, col = i // 4 + 1, i % 4 + 1
                
                for dataset, color in [("era5", "#5D1D95"), ("merra2", "#5DA9E9")]:
                    data = month_data[month_data["Dataset"] == dataset]
                    if not data.empty:
                        fig.add_trace(go.Scatter(
                            x=data["Global_Temp"], y=data["Average_Temperature"],
                            mode="markers", marker=dict(color=color, size=4),
                            name=f"{dataset.upper()} Historical",
                            legendgroup=f"{dataset.upper()}_historical",
                            showlegend=(i == 0)
                        ), row=row, col=col)
        
        # Add future projections for the specified scenario only
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        scenario_colors = {
            "aa": "rgba(36, 161, 72, 0.3)",  # Green for Accelerated Actions
            "ct": "rgba(255, 131, 137, 0.3)"  # Red for Current Trends
        }
        scenario_names = {
            "aa": "Accelerated Actions",
            "ct": "Current Trends"
        }
        
        if self.scenario in scenario_colors:
            for i, month in enumerate(months):
                row, col = i // 4 + 1, i % 4 + 1
                
                self._add_projection_bands(fig, region, month, self.scenario, 
                                         scenario_colors[self.scenario], 
                                         scenario_names[self.scenario], row, col)
        
        fig.update_layout(
            title=f"Historical Data and Future Projections by Global Temperature: {region}",
            height=800, width=1200,
        )
        
        return fig

    def make_by_year_plot(self, region):
        """
        Plot historical temperature data and future projections by year.
        X-axis: Year, Y-axis: Local Temperature
        Shows scatter plots (no regression lines) plus projection bands.
        """
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = make_subplots(rows=3, cols=4, subplot_titles=months, vertical_spacing=0.05, shared_xaxes=True)
        
        # Plot historical data as scatter plots
        historical_df = self.data[self.data["Region"] == region].copy()
        for i, month in enumerate(months):
            month_data = historical_df[historical_df["Month"] == month]
            row, col = i // 4 + 1, i % 4 + 1
            
            # Add historical scatter for both datasets
            for dataset, color in [("era5", "#5D1D95"), ("merra2", "#5DA9E9")]:
                data = month_data[month_data["Dataset"] == dataset]
                if not data.empty:
                    fig.add_trace(go.Scatter(
                        x=data["Year"], y=data["Average_Temperature"],
                        mode="markers", marker=dict(color=color, size=4),
                        name=f"{dataset.upper()} Historical",
                        legendgroup=f"{dataset.upper()}_historical",
                        showlegend=(i == 0)
                    ), row=row, col=col)
        
        # Add future projections for the specified scenario
        scenario_colors = {
            "aa": "rgba(36, 161, 72, 0.3)",  # Green for Accelerated Actions
            "ct": "rgba(255, 131, 137, 0.3)"  # Red for Current Trends
        }
        scenario_names = {
            "aa": "Accelerated Actions",
            "ct": "Current Trends"
        }
        
        if self.scenario in scenario_colors:
            for i, month in enumerate(months):
                row, col = i // 4 + 1, i % 4 + 1
                self._add_year_projection_bands(fig, region, month, self.scenario,
                                              scenario_colors[self.scenario],
                                              scenario_names[self.scenario], row, col)
        
        fig.update_layout(
            template="simple_white",  # White background
            title=dict(text=f"Historical Data and Future Projections by Year: {region}", 
                       ),
            height=800, width=1200,  # 400-100, 900-100
            margin=dict(l=60, r=60, t=60, b=80),  # Add margins to prevent cutoff
        )
        
        # Add shared axis labels in the middle of the entire plot area
        fig.add_annotation(
            text="Year",
            xref="paper", yref="paper",
            x=0.5, y=-0.08,  # Middle of plot area, below subplots
            showarrow=False,
            font=dict(size=14, color="black"),  # Smaller font for smaller plot
            xanchor="center"
        )
        
        fig.add_annotation(
            text="Regional Temperature (°C)",
            xref="paper", yref="paper",
            x=-0.04, y=0.5,  # Left of plot area, middle height
            showarrow=False,
            font=dict(size=14, color="black"),  # Smaller font for smaller plot
            textangle=-90,  # Rotate 90 degrees
            xanchor="center", yanchor="middle"
        )
        
        return fig

    def _add_year_projection_bands(self, fig, region: str, month: str, scenario: str,
                                  color: str, name: str, row: int, col: int):
        """Add 95% prediction interval bands for future projections with year on x-axis."""
        # Load scenario temperature projections
        temp_df = self._load_scenario_temps(scenario)
        if temp_df.empty:
            return
            
        # Get bootstrap coefficients
        ints, slps, sigma = self._get_bootstrap_coeffs(region, month)
        
        # Calculate centering mean from historical data
        hist_data = self.data[(self.data["Region"] == region) & (self.data["Month"] == month)]
        center_mean = hist_data["Global_Temp"].mean() if not hist_data.empty else 0.0
        
        # For each year, calculate prediction intervals
        years = temp_df.iloc[:, 0].values
        y_lowers = []
        y_uppers = []
        y_medians = []
        
        for year in years:
            year_temps = temp_df[temp_df.iloc[:, 0] == year].iloc[:, 1:].values.flatten()
            year_temps = year_temps[~np.isnan(year_temps)]
            
            if len(year_temps) > 0:
                # Monte Carlo for prediction interval
                n_samples = 1000
                rng = np.random.default_rng(123)
                
                # Sample from bootstrap coefficients
                boot_idx = rng.integers(0, len(ints), n_samples)
                sampled_ints = ints[boot_idx]
                sampled_slps = slps[boot_idx]
                
                # Sample from predictor uncertainty
                temp_samples = rng.choice(year_temps, n_samples)
                
                # Calculate predictions with all uncertainties
                y_pred = sampled_ints + sampled_slps * (temp_samples - center_mean)
                y_pred += rng.normal(0, sigma, n_samples)  # Add residual noise
                
                # Calculate prediction intervals and median
                y_lower, y_upper = np.percentile(y_pred, [2.5, 97.5])
                y_median = np.median(y_pred)
                
                y_lowers.append(y_lower)
                y_uppers.append(y_upper)
                y_medians.append(y_median)
        
        if years.size > 0:
            # Add prediction band
            fig.add_trace(go.Scatter(
                x=np.concatenate([years, years[::-1]]),
                y=np.concatenate([y_uppers, y_lowers[::-1]]),
                fill='toself', fillcolor=color, opacity=0.3,
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip', showlegend=(row == 1 and col == 1),
                name=f"{name} 95% PI",
                legendgroup=f"{scenario}_projection"
            ), row=row, col=col)
            
            # Add median projection line
            fig.add_trace(go.Scatter(
                x=years, y=y_medians, mode='lines+markers',
                line=dict(color=color.replace('rgba', 'rgb').replace(', 0.3', ''), width=2),
                marker=dict(size=3),
                name=f"{name} Median", showlegend=(row == 1 and col == 1),
                legendgroup=f"{scenario}_projection"
            ), row=row, col=col)
        
    def make_plots(self, region):
        temp_fig = self.make_by_temp_plot(region = region)
        year_fig = self.make_by_year_plot(region = region)
        return temp_fig, year_fig

    def pregenerate_all_plots(self, output_dir="static_plots", variables=None):
        """
        Pre-generate state/scenario/variable combinations as static HTML files for webapp.
        This minimizes compute time and enables ~1s loading.
        
        Args:
            output_dir: Directory to save plots
            variables: List of variables to generate. If None, generates for all available variables.
        """
        import os
        from pathlib import Path
        import json
        from datetime import datetime
        
        # Create output directory structure
        base_path = Path(output_dir)
        base_path.mkdir(exist_ok=True)
        
        # Get all states from the data
        states = self.data["Region"].unique()
        scenarios = ["aa", "ct"]
        
        # Use specified variables or default to all
        if variables is None:
            variables = ["T2MMAX", "T2MMEAN", "T2MMIN"]
        
        var_folders = {"T2MMAX": "Max", "T2MMEAN": "Mean", "T2MMIN": "Min"}
        
        # Track generation metadata
        metadata = {
            "generated_at": datetime.now().isoformat(),
            "total_plots": len(states) * len(scenarios) * len(variables) * 2,  # 2 plot types per state/scenario/variable
            "states": states.tolist(),
            "scenarios": scenarios,
            "variables": variables,
            "plot_types": ["temp", "year"]
        }
        
        print(f"Pre-generating {metadata['total_plots']} plots for {len(states)} states × {len(scenarios)} scenarios × {len(variables)} variables...")
        print(f"Variables to generate: {', '.join(variables)}")
        
        # Generate plots for each combination
        for i, state in enumerate(states):
            print(f"Processing state {i+1}/{len(states)}: {state}")
            
            for scenario in scenarios:
                # Create scenario-specific directory
                scenario_dir = base_path / scenario
                scenario_dir.mkdir(exist_ok=True)
                
                for variable in variables:
                    # Create variable-specific subdirectory
                    var_dir = scenario_dir / var_folders[variable]
                    var_dir.mkdir(exist_ok=True)
                    
                    try:
                        # Create instance for this scenario and variable
                        app_instance = AppFunctionsforPooledData(scenario=scenario, var=variable)
                        
                        # Generate temperature plot
                        temp_fig = app_instance.make_by_temp_plot(region=state)
                        temp_file = var_dir / f"{state}_temp.html"
                        temp_fig.write_html(
                            str(temp_file),
                            include_plotlyjs='cdn',  # Use CDN for smaller files
                        )
                        
                        # Generate year plot
                        year_fig = app_instance.make_by_year_plot(region=state)
                        year_file = var_dir / f"{state}_year.html"
                        year_fig.write_html(
                            str(year_file),
                            include_plotlyjs='cdn',
                        )
                        
                    except Exception as e:
                        print(f"Error generating plots for {state}/{scenario}/{variable}: {e}")
                        continue
        
        # Save metadata
        with open(base_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Pre-generation complete! Files saved in '{output_dir}/'")
        print(f"📁 Directory structure:")
        print(f"   {output_dir}/")
        print(f"   ├── aa/")
        print(f"   │   ├── MA_temp.html")
        print(f"   │   ├── MA_year.html")
        print(f"   │   └── ...")
        print(f"   ├── ct/")
        print(f"   │   ├── MA_temp.html") 
        print(f"   │   └── ...")
        print(f"   └── metadata.json")
        
        return metadata

    def get_plot_path(state, scenario, plot_type, variable="T2MMAX", base_dir="static_plots"):
        """
        Get the file path for a specific plot combination.
        
        Args:
            state: State code (e.g., 'MA')
            scenario: 'aa' or 'ct'
            plot_type: 'temp' or 'year'
            variable: 'T2MMAX', 'T2MMEAN', or 'T2MMIN'
            base_dir: Base directory for static plots
        """
        var_folders = {"T2MMAX": "Max", "T2MMEAN": "Mean", "T2MMIN": "Min"}
        var_folder = var_folders.get(variable, "Max")
        return f"{base_dir}/{scenario}/{var_folder}/{state}_{plot_type}.html"

    def get_r2(self, region: str, month: str, dataset: str = {"era5", "merra2"}) -> float:
        """
        Calculate R² (coefficient of determination) for a specific region and month
        using historical data and regression parameters.
        
        Parameters
        ----------
        region : str
            Region code (e.g., 'MA', 'CA')
        month : str
            Month name (e.g., 'Jan', 'Feb')
        dataset : str, optional
            Dataset to use ('era5', 'merra2', or None for pooled)
            If None, uses pooled regression parameters
        
        Returns
        -------
        float
            R² value (coefficient of determination)
        """
        if dataset == "era5":
            hist_data = self.get_era5_historical_data(region = region)
        elif dataset == "merra2":
            hist_data = self.get_merra2_historical_data(region = region)
        
        hist_data = hist_data[hist_data["Month"] == month]
        
        # Get regression parameters
        qi = self.uncertainty_intervals[(self.uncertainty_intervals["Region"] == region) & 
                                       (self.uncertainty_intervals["Month"] == month)]
        
        if qi.empty:
            return np.nan
        
        # Use median parameters (50th percentile)
        intercept = qi["Intercept_50th"].iloc[0]
        slope = qi["Slope_50th"].iloc[0]
        
        # Get global temperature data and calculate centering mean
        global_temps = hist_data["Global_Temp"].values
        center_mean = global_temps.mean()
        
        # Calculate centered global temperature (predictor)
        x_centered = global_temps - center_mean
        
        # Get observed regional temperatures (response)
        y_observed = hist_data["Average_Temperature"].values
        
        # Calculate predicted values using regression equation
        y_predicted = intercept + slope * x_centered
        
        # Calculate R²
        r = pearsonr(y_observed, y_predicted)
        r2 = r[0] ** 2
        
        return r2
    
    def plot_r2(self):
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        regions = self.data["Region"].unique()
        r2_values = []
        for month in months:
            for region in regions:
                era5_r2 = self.get_r2(region = region, month = month, dataset = "era5")
                merra2_r2 = self.get_r2(region = region, month = month, dataset = "merra2")
                r2 = (era5_r2 + merra2_r2) / 2
                r2_values.append({"Region": region, "Month": month, "r2": r2})
        df = pd.DataFrame(r2_values)
        fig = px.choropleth(df,
                            locations="Region", 
                            locationmode="USA-states", 
                            color="r2", 
                            facet_col="Month",
                            facet_col_wrap=4,
                            facet_col_spacing=0,
                            color_continuous_scale="Greens",
                            scope="usa")
        fig.update_layout(
            title="R² Validations",
            coloraxis_colorbar=dict(
                title="R²",
                x=1.0,
                y=0.5
            ),
            width=1500,
            height=1000
        )
        return fig

class PooledEstimator:
    """
    Estimate temperature-global-mean regressions for ERA5 & MERRA2
    and produce an inverse-variance-pooled result with bootstrap SEs.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        ['Year', 'Month', 'Average_Temperature',
         'Dataset' ('era5'|'merra2'), 'Global_Temp', 'Region'].
    n_boot : int, default 500
        Number of bootstrap replicates for pooled estimates.
    seed : int, default 123
        RNG seed for reproducibility.
    """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def __init__(self,
                 df:     pd.DataFrame,
                 n_boot: int = 500,
                 seed:   int = 123):
        """
        df must contain columns:
          Year, Month, Average_Temperature, Dataset, Global_Temp, Region
        """
        # restrict to common period
        df = df[df["Year"].between(1980, 2022)].copy()

        # dataset‑specific centring of predictor
        means = df.groupby("Dataset")["Global_Temp"].transform("mean")
        df["Global_Temp_c"] = df["Global_Temp"] - means

        self.df      = df
        self.n_boot  = n_boot
        self.rng     = np.random.default_rng(seed)
        self.months  = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def save_dataset_regressions(self, outfile: str) -> None:
        """Save ERA5 and MERRA2 regressions separately to CSV."""
        res = self._dataset_regressions()
        res.to_csv(outfile, index=False)

    def save_pooled_bootstrap(self, outfile: str) -> None:
        """Save pooled bootstrap regressions (slope & intercept) to CSV."""
        res = self._pooled_bootstrap()
        res.to_csv(outfile, index=False)

    def save_uncertainty_intervals(self, outfile: str) -> None:
        """Save uncertainty intervals (5th, 50th, 95th percentiles) to CSV."""
        res = self._produce_uncertainty_intervals()
        res.to_csv(outfile, index=False)

    @staticmethod
    def load_results(path: str) -> pd.DataFrame:
        """Load a previously saved CSV into a DataFrame."""
        return pd.read_csv(path)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _fast_ols(x: np.ndarray, y: np.ndarray):
        """
        Simple linear regression y = a + b x.
        Returns (intercept, slope, se_intercept, se_slope).
        """
        n = len(x)
        if n < 3:
            return np.nan, np.nan, np.nan, np.nan

        x_mean = x.mean()
        y_mean = y.mean()
        x_c = x - x_mean
        y_c = y - y_mean
        Sxx = np.dot(x_c, x_c)
        if Sxx == 0:
            return np.nan, np.nan, np.nan, np.nan

        slope = np.dot(x_c, y_c) / Sxx
        intercept = y_mean - slope * x_mean

        resid = y - (intercept + slope * x)
        sigma2 = np.dot(resid, resid) / (n - 2)

        se_slope = np.sqrt(sigma2 / Sxx)
        se_int = np.sqrt(sigma2 * (1 / n + x_mean**2 / Sxx))
        return intercept, slope, se_int, se_slope

    @staticmethod
    def _wald_p(est, se):
        if np.isnan(est) or np.isnan(se) or se == 0:
            return np.nan
        z = np.abs(est / se)
        return 2 * (1 - norm.cdf(z))

    # ------------------------------------------------------------------
    # step 1 – individual dataset regressions
    # ------------------------------------------------------------------
    def _dataset_regressions(self):
        rows = []
        grp_cols = ["Region", "Month", "Dataset"]
        for (reg, mon, dat), sub in self.df.groupby(grp_cols):
            x = sub["Global_Temp_c"].values
            y = sub["Average_Temperature"].values
            _, slope, _, se_slope = self._fast_ols(x, y)
            rows.append(dict(Region=reg,
                             Month=mon,
                             Dataset=dat,
                             Slope=slope,
                             Slope_SE=se_slope,
                             Slope_p=self._wald_p(slope, se_slope)))
        return (pd.DataFrame(rows)
                .sort_values(["Region", "Month", "Dataset"])
                .reset_index(drop=True))

    # ------------------------------------------------------------------
    # step 2 – pooled bootstrap
    # ------------------------------------------------------------------
    def _pooled_bootstrap(self):
        regions = self.df["Region"].unique()

        pooled_rows = []

        for reg in regions:
            df_reg = self.df[self.df["Region"] == reg]
            for mon in self.months:
                df_rm = df_reg[df_reg["Month"] == mon]
                era5 = df_rm[df_rm["Dataset"] == "era5"]
                mer  = df_rm[df_rm["Dataset"] == "merra2"]
                yrs = np.intersect1d(era5["Year"], mer["Year"])
                if len(yrs) < 3:
                    continue

                era5 = era5.set_index("Year").loc[yrs]
                mer  = mer.set_index("Year").loc[yrs]
                x = era5["Global_Temp_c"].values      # aligned predictor
                y_e = era5["Average_Temperature"].values
                y_m = mer["Average_Temperature"].values
                L = len(x)

                ints = np.empty(self.n_boot)
                slps = np.empty(self.n_boot)

                for b in range(self.n_boot):
                    idx = self.rng.integers(0, L, size=L)
                    int_e, sl_e, se_ie, se_se = self._fast_ols(x[idx], y_e[idx])
                    int_m, sl_m, se_im, se_sm = self._fast_ols(x[idx], y_m[idx])

                    # invalid replicate → mark NaN
                    if any(np.isnan([se_ie, se_im, se_se, se_sm])) \
                       or any(s == 0 for s in [se_ie, se_im, se_se, se_sm]):
                        ints[b] = np.nan
                        slps[b] = np.nan
                        continue

                    # inverse‑variance weighting
                    w_ie, w_im = 1 / se_ie**2, 1 / se_im**2
                    w_se, w_sm = 1 / se_se**2, 1 / se_sm**2

                    ints[b] = (w_ie * int_e + w_im * int_m) / (w_ie + w_im)
                    slps[b] = (w_se * sl_e  + w_sm * sl_m) / (w_se + w_sm)

                ints = ints[~np.isnan(ints)]
                slps = slps[~np.isnan(slps)]
                if len(ints) == 0:
                    continue

                int_mean = ints.mean()
                slp_mean = slps.mean()
                int_se   = ints.std(ddof=1)
                slp_se   = slps.std(ddof=1)

                pooled_rows.append(dict(
                    Region=reg, Month=mon,
                    Pooled_Intercept=int_mean, Intercept_SE=int_se,
                    Pooled_Slope=slp_mean,     Slope_SE=slp_se,
                    Slope_p=self._wald_p(slp_mean, slp_se)
                ))

        return (pd.DataFrame(pooled_rows)
                .sort_values(["Region", "Month"])
                .reset_index(drop=True))

    def _pooled_residual_std(self, x: np.ndarray, y: np.ndarray, beta0: float, beta1: float) -> float:
        """Return pooled residual standard deviation s given one line."""
        y_hat = beta0 + beta1 * x
        resid = y - y_hat
        n = len(x)
        return np.sqrt(np.sum(resid ** 2) / (n - 2))

    def _analytic_bands(self, x: np.ndarray, y: np.ndarray, beta0: float, beta1: float, alpha=0.05):
        """Return x_grid, mean line, 100*(1-alpha)% CI & PI bands using textbook formulas."""
        n = len(x)
        x_bar = x.mean()
        ssx = np.sum((x - x_bar) ** 2)
        s = self._pooled_residual_std(x, y, beta0, beta1)

        # Grid for smooth plotting
        x_grid = np.linspace(x.min(), x.max(), 400)
        y_hat = beta0 + beta1 * x_grid

        se_mean = s * np.sqrt(1 / n + (x_grid - x_bar) ** 2 / ssx)  # CI half‑widths
        se_pred = s * np.sqrt(1 + 1 / n + (x_grid - x_bar) ** 2 / ssx)  # PI half‑widths

        tcrit = t.ppf(1 - alpha / 2, df=n - 2)

        half_ci = tcrit * se_mean
        half_pi = tcrit * se_pred
        return x_grid, y_hat, half_ci, half_pi

    def _bootstrap_bands(self, x_grid: np.ndarray, ints: np.ndarray, slps: np.ndarray, resid_std: float | None = None, alpha=0.05, rng=np.random.default_rng()):
        """Return 100*(1-alpha)% bootstrap percentile bands.

        If resid_std is supplied, create a **prediction** band by adding Gaussian noise with that
        std to each bootstrap line before taking percentiles.  If resid_std is None, a **confidence**
        band for the mean response is returned.
        """
        lines = slps[:, None] * x_grid + ints[:, None]  # shape (B, G)
        if resid_std is not None:
            noise = rng.normal(0, resid_std, size=lines.shape)
            lines = lines + noise  # prediction band

        lo, hi = np.percentile(lines, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
        return lo, hi

    def _produce_uncertainty_intervals(self):
        """
        Generate uncertainty bounds for regression coefficients based on bootstrap.
        Returns a DataFrame with 5th, 50th, and 95th percentiles for slopes and intercepts.
        """
        years = self.df["Year"].unique()
        regions = self.df["Region"].unique()
        
        uncertainty_rows = []
        
        for reg in regions:
            df_reg = self.df[self.df["Region"] == reg]
            for mon in self.months:
                df_rm = df_reg[df_reg["Month"] == mon]
                era5 = df_rm[df_rm["Dataset"] == "era5"]
                mer  = df_rm[df_rm["Dataset"] == "merra2"]
                yrs = np.intersect1d(era5["Year"], mer["Year"])
                if len(yrs) < 3:
                    continue
                
                era5 = era5.set_index("Year").loc[yrs]
                mer  = mer .set_index("Year").loc[yrs]
                x = era5["Global_Temp_c"].values      # aligned predictor
                y_e = era5["Average_Temperature"].values
                y_m = mer ["Average_Temperature"].values
                L = len(x)
                
                # Store all bootstrap samples
                ints = np.empty(self.n_boot)
                slps = np.empty(self.n_boot)
                
                for b in range(self.n_boot):
                    idx = self.rng.integers(0, L, size=L)
                    int_e, sl_e, se_ie, se_se = self._fast_ols(x[idx], y_e[idx])
                    int_m, sl_m, se_im, se_sm = self._fast_ols(x[idx], y_m[idx])
                    
                    # invalid replicate → mark NaN
                    if any(np.isnan([se_ie, se_im, se_se, se_sm])) \
                       or any(s == 0 for s in [se_ie, se_im, se_se, se_sm]):
                        ints[b] = np.nan
                        slps[b] = np.nan
                        continue
                    
                    # inverse‑variance weighting
                    w_ie, w_im = 1 / se_ie**2, 1 / se_im**2
                    w_se, w_sm = 1 / se_se**2, 1 / se_sm**2
                    
                    ints[b] = (w_ie * int_e + w_im * int_m) / (w_ie + w_im)
                    slps[b] = (w_se * sl_e  + w_sm * sl_m) / (w_se + w_sm)
                
                # Remove NaN values
                ints = ints[~np.isnan(ints)]
                slps = slps[~np.isnan(slps)]
                if len(ints) == 0:
                    continue

                sigma = self._pooled_residual_std(x, y_e, np.median(ints), np.median(slps))
                cols = ["state", "month", "member", "intercept", "slope", "resid_std"]
                for k, (b0, b1) in enumerate(zip(ints, slps)):
                    uncertainty_rows.append((reg, mon, k, b0, b1, sigma))
                                
        return (pd.DataFrame(uncertainty_rows, columns=cols)
                .sort_values(["state", "month", "member"])
                .reset_index(drop=True))

    def get_uncertainty_intervals_for_region_month(self, region: str, month: str) -> dict:
        """
        Get uncertainty intervals for a specific region and month.
        Returns a dict with slope and intercept percentiles.
        """
        uncertainty_df = self._produce_uncertainty_intervals()
        subset = uncertainty_df[(uncertainty_df['state'] == region) & 
                               (uncertainty_df['month'] == month)]
        
        if subset.empty:
            return None
        
        # Calculate percentiles from bootstrap samples
        slopes = subset['slope'].values
        intercepts = subset['intercept'].values
        
        return {
            'slope_5th': np.percentile(slopes, 5),
            'slope_50th': np.percentile(slopes, 50), 
            'slope_95th': np.percentile(slopes, 95),
            'intercept_5th': np.percentile(intercepts, 5),
            'intercept_50th': np.percentile(intercepts, 50),
            'intercept_95th': np.percentile(intercepts, 95)
        }

class UncertaintyIntervalLoader:
    """
    Utility class for loading and accessing uncertainty intervals from CSV.
    Optimized for fast lookups by region and month.
    """
    
    def __init__(self, csv_path: str):
        """
        Load uncertainty intervals from CSV file.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file created by PooledEstimator.save_uncertainty_intervals()
        """
        self.df = pd.read_csv(csv_path)
        # Create a multi-index for fast lookups
        self.df = self.df.set_index(['state', 'month'])
    
    def get_intervals(self, region: str, month: str) -> dict:
        """
        Get uncertainty intervals for a specific region and month.
        
        Parameters
        ----------
        region : str
            Region code (e.g., 'MA', 'CA')
        month : str
            Month name (e.g., 'Jan', 'Feb')
            
        Returns
        -------
        dict or None
            Dictionary with slope and intercept percentiles, or None if not found
        """
        try:
            subset = self.df.loc[(region, month)]
            # Calculate percentiles from bootstrap samples
            slopes = subset['slope'].values
            intercepts = subset['intercept'].values
            
            return {
                'slope_5th': np.percentile(slopes, 5),
                'slope_50th': np.percentile(slopes, 50), 
                'slope_95th': np.percentile(slopes, 95),
                'intercept_5th': np.percentile(intercepts, 5),
                'intercept_50th': np.percentile(intercepts, 50),
                'intercept_95th': np.percentile(intercepts, 95)
            }
        except KeyError:
            return None
    
    def get_all_regions(self) -> list:
        """Get list of all available regions."""
        return self.df.index.get_level_values('state').unique().tolist()
    
    def get_all_months(self) -> list:
        """Get list of all available months."""
        return self.df.index.get_level_values('month').unique().tolist()
    
    def get_slopes_for_region(self, region: str) -> pd.DataFrame:
        """
        Get all slope uncertainty intervals for a specific region.
        
        Parameters
        ----------
        region : str
            Region code
            
        Returns
        -------
        pd.DataFrame
            DataFrame with months and slope percentiles
        """
        try:
            region_data = self.df.loc[region]
            # Calculate percentiles for each month
            results = []
            for month in region_data.index.get_level_values('month').unique():
                month_data = region_data.loc[month]
                slopes = month_data['slope'].values
                results.append({
                    'month': month,
                    'slope_5th': np.percentile(slopes, 5),
                    'slope_50th': np.percentile(slopes, 50),
                    'slope_95th': np.percentile(slopes, 95)
                })
            return pd.DataFrame(results)
        except KeyError:
            return pd.DataFrame()

class PlotlySlopeMap:
    """Visualise bootstrap regression bands for US statemonth pairs.

    The class now supports **two modes** automatically:

    1. **Per‑dataset** mode – when the coefficient dataframe *has* a
       `dataset` column (e.g., ERA5 vs MERRA2).  The figure overlays one band
       per dataset.
    2. **Pooled** mode – when `dataset` is *absent*.  The coefficients are
       treated as coming from a single, already‑pooled model and only one band
       is drawn.

    Parameters
    ----------
    coeff_df : pandas.DataFrame
        Long‑format bootstrap coefficient table with columns
        `state, month, member, intercept, slope, resid_std` and *optionally*
        `dataset`.
    hist_df : pandas.DataFrame
        Historical observations table (kept for possible scatter overlays).
    alpha : float, default 0.05
        Tail probability → 1‑alpha is the nominal coverage (0.05 → 95 %).
    x_range : (float, float) | None, default None
        Range of the global‑temperature x‑axis.  Defaults to the min/max of
        `hist_df['Global_Temp']`.
    """

    _MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    _DEFAULT_COLORS = ["royalblue", "orangered", "seagreen", "mediumpurple"]

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------
    def __init__(self, *, coeff_df: pd.DataFrame, hist_df: pd.DataFrame,
                 alpha: float = 0.05, x_range: Optional[Tuple[float, float]] = None):
        self.alpha = alpha
        self.coeff_df = coeff_df.copy()
        self.hist_df = hist_df.copy()

        # -----   Validate required columns   -----
        base_cols = {"state", "month", "member", "intercept", "slope", "resid_std"}
        if base_cols - set(self.coeff_df.columns):
            raise ValueError(f"coeff_df missing columns: {base_cols - set(self.coeff_df.columns)}")

        # detect whether we have per‑dataset coefficients or a pooled model
        self.per_dataset = "dataset" in self.coeff_df.columns
        if self.per_dataset:
            self.datasets = sorted(self.coeff_df["dataset"].unique().tolist())
        else:
            # Always use pooled model for regression bands
            self.datasets = ["Pooled"]
            self.coeff_df["dataset"] = "Pooled"  # makes queries uniform

        # Calculate dataset-specific global temperature means for centering
        self.dataset_means = {}
        for dataset in self.datasets:
            if dataset == "Pooled":
                # For pooled model, use the overall mean
                self.dataset_means[dataset] = self.hist_df["Global_Temp"].mean()
            else:
                # For per-dataset model, use dataset-specific mean
                dataset_data = self.hist_df[self.hist_df["Dataset"].str.lower() == dataset.lower()]
                if not dataset_data.empty:
                    self.dataset_means[dataset] = dataset_data["Global_Temp"].mean()
                else:
                    # Fallback to overall mean if dataset not found
                    self.dataset_means[dataset] = self.hist_df["Global_Temp"].mean()

        # colour maps (cycled if more than 4 datasets)
        self._LINE_COLOUR: Dict[str, str] = {}
        self._BAND_COLOUR: Dict[str, str] = {}
        for idx, ds in enumerate(self.datasets):
            base = self._DEFAULT_COLORS[idx % len(self._DEFAULT_COLORS)]
            self._LINE_COLOUR[ds] = base
            # 30 % opacity fill - convert to rgba format
            if base.startswith("rgb"):
                self._BAND_COLOUR[ds] = base.replace("rgb", "rgba").replace(")", ",0.30)")
            else:
                # Simple color to rgba conversion for common colors
                color_map = {
                    "royalblue": "rgba(65, 105, 225, 0.30)",
                    "orangered": "rgba(255, 69, 0, 0.30)", 
                    "seagreen": "rgba(46, 139, 87, 0.30)",
                    "mediumpurple": "rgba(147, 112, 219, 0.30)"
                }
                self._BAND_COLOUR[ds] = color_map.get(base, f"rgba(0, 0, 0, 0.30)")

        # Use historical data's Global_Temperature range for x-axis bounds
        if x_range is None:
            if {"Global_Temp"}.issubset(self.hist_df.columns):
                self.x_range = (self.hist_df["Global_Temp"].min(),
                                self.hist_df["Global_Temp"].max())
            else:
                self.x_range = (0, 1)
        else:
            self.x_range = x_range

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _subset_coeff(self, *, state: str, month: str, dataset: str):
        sub = self.coeff_df.query("state == @state and month == @month and dataset == @dataset")
        if sub.empty:
            raise ValueError(f"No coefficients for state={state}, month={month}, dataset={dataset}")
        return (sub["intercept"].to_numpy(),
                sub["slope"].to_numpy(),
                sub["resid_std"].iloc[0])

    @staticmethod
    def _bootstrap_band(x_grid: np.ndarray, ints: np.ndarray, slps: np.ndarray,
                        resid_std: Optional[float], alpha: float):
        lines = slps[:, None] * x_grid + ints[:, None]
        if resid_std is not None:
            lines += np.random.default_rng().normal(0, resid_std, size=lines.shape)
        lo, hi = np.percentile(lines, [100*alpha/2, 100*(1-alpha/2)], axis=0)
        return lo, hi

    def _x_grid(self, n: int = 400):
        return np.linspace(*self.x_range, n)

    # ------------------------------------------------------------------
    # Single‑month figure
    # ------------------------------------------------------------------
    def _single_panel(self, *, state: str, month: str, prediction: bool):
        x_grid = self._x_grid()
        fig = go.Figure()

        # Add regression bands (always use pooled model)
        for ds in self.datasets:
            ints, slps, sigma = self._subset_coeff(state=state, month=month, dataset=ds)
            
            # Center the x-grid values for the regression calculation
            x_centered = x_grid - self.dataset_means[ds]
            
            # Calculate regression lines using centered x-values
            lo, hi = self._bootstrap_band(x_centered, ints, slps,
                                          resid_std=(sigma if prediction else None),
                                          alpha=self.alpha)
            y_hat = np.median(ints) + np.median(slps) * x_centered

            # Plot regression bands
            band_type = "PI" if prediction else "CI"
            fig.add_trace(go.Scatter(x=x_grid, y=y_hat, mode="lines",
                                     name=f"{ds} median", 
                                     legendgroup=f"{ds}_median",
                                     line=dict(color=self._LINE_COLOUR[ds])))
            fig.add_trace(go.Scatter(x=np.concatenate([x_grid, x_grid[::-1]]),
                                     y=np.concatenate([lo, hi[::-1]]), fill="toself",
                                     fillcolor=self._BAND_COLOUR[ds], line=dict(color="rgba(255,255,255,0)"),
                                     hoverinfo="skip", name=f"{ds} {band_type}",
                                     legendgroup=f"{ds}_{band_type.lower()}"))

        # Always add historical data for both datasets
        for dataset_name in ["era5", "merra2"]:
            dataset_data = self.hist_df[(self.hist_df["Dataset"].str.lower() == dataset_name) & 
                                       (self.hist_df["Region"] == state) & 
                                       (self.hist_df["Month"] == month)]
            
            if not dataset_data.empty:
                # Use different colors for each dataset
                colors = {"era5": "#5D1D95", "merra2": "#5DA9E9"}
                fig.add_trace(go.Scatter(
                    x=dataset_data["Global_Temp"], 
                    y=dataset_data["Average_Temperature"],
                    mode="markers", 
                    name=f"{dataset_name.upper()} Historical",
                    legendgroup=f"{dataset_name.upper()}_historical",
                    marker=dict(color=colors[dataset_name], size=6),
                    showlegend=True
                ))

        kind = "prediction" if prediction else "confidence"
        title = f"{state} – {month}: 95 % {kind} band with historical data"
        fig.update_layout(template="simple_white",
                          title=title,
                          xaxis_title="Global mean temperature (°C)",
                          yaxis_title="State‑average temperature (°C)")
        return fig

    # Public API
    # ------------------------------------------------------------------
    def create_confidence_plot(self, *, state: str, month: str):
        return self._single_panel(state=state, month=month, prediction=False)

    def create_prediction_plot(self, *, state: str, month: str):
        return self._single_panel(state=state, month=month, prediction=True)

    # ------------------------------------------------------------------
    # Grid of 4×3 months
    # ------------------------------------------------------------------
    def _grid(self, *, state: str, prediction: bool):
        x_grid = self._x_grid()
        fig = make_subplots(rows=3, cols=4, subplot_titles=self._MONTHS,
                            shared_xaxes=True, shared_yaxes=False, vertical_spacing = 0.05)

        # Show legend only once per dataset / band type
        showleg_line = {ds: True for ds in self.datasets}
        showleg_band = {ds: True for ds in self.datasets}
        showleg_era5 = True
        showleg_merra2 = True

        for idx, month in enumerate(self._MONTHS, start=1):
            r = (idx - 1)//4 + 1
            c = (idx - 1)%4 + 1
            
            # Add regression bands (always use pooled model)
            for ds in self.datasets:
                ints, slps, sigma = self._subset_coeff(state=state, month=month, dataset=ds)
                
                # Center the x-grid values for the regression calculation
                x_centered = x_grid - self.dataset_means[ds]
                
                # Calculate regression lines using centered x-values
                lo, hi = self._bootstrap_band(x_centered, ints, slps,
                                               resid_std=(sigma if prediction else None),
                                               alpha=self.alpha)
                y_hat = np.median(ints) + np.median(slps) * x_centered

                # Plot regression bands
                band_type = "PI" if prediction else "CI"
                fig.add_trace(go.Scatter(x=x_grid, y=y_hat, mode="lines",
                                          line=dict(color=self._LINE_COLOUR[ds]),
                                          name=f"{ds} median" if showleg_line[ds] else None,
                                          legendgroup=f"{ds}_median",
                                          showlegend=showleg_line[ds]),
                               row=r, col=c)
                fig.add_trace(go.Scatter(x=np.concatenate([x_grid, x_grid[::-1]]),
                                          y=np.concatenate([lo, hi[::-1]]), fill="toself",
                                          fillcolor=self._BAND_COLOUR[ds],
                                          line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
                                          name=f"{ds} {band_type}" if showleg_band[ds] else None,
                                          legendgroup=f"{ds}_{band_type.lower()}",
                                          showlegend=showleg_band[ds]),
                               row=r, col=c)
                showleg_line[ds] = False
                showleg_band[ds] = False

            # Always add historical data for both datasets
            for dataset_name in ["era5", "merra2"]:
                dataset_data = self.hist_df[(self.hist_df["Dataset"].str.lower() == dataset_name) & 
                                           (self.hist_df["Region"] == state) & 
                                           (self.hist_df["Month"] == month)]
                
                if not dataset_data.empty:
                    # Use different colors for each dataset
                    colors = {"era5": "#5D1D95", "merra2": "#5DA9E9"}
                    show_legend = showleg_era5 if dataset_name == "era5" else showleg_merra2
                    fig.add_trace(go.Scatter(
                        x=dataset_data["Global_Temp"], 
                        y=dataset_data["Average_Temperature"],
                        mode="markers", 
                        name=f"{dataset_name.upper()} Historical" if show_legend else None,
                        legendgroup=f"{dataset_name.upper()}_historical",
                        marker=dict(color=colors[dataset_name], size=4),
                        showlegend=show_legend
                    ), row=r, col=c)
                    
                    if dataset_name == "era5":
                        showleg_era5 = False
                    else:
                        showleg_merra2 = False

        kind = "prediction" if prediction else "confidence"
        title = f"{state}: 95 % {kind} band with historical data"
        fig.update_layout(template="simple_white", height=800, width=1100,
                          title=dict(text=title, x=0.5))
        # Axis labels on outer edge
        for r in range(1, 5):
            fig.update_xaxes(title_text="Global Temp (°C)" if r == 4 else None, row=r, col=1)
        for c in range(1, 4):
            fig.update_yaxes(title_text="State Temp (°C)" if c == 1 else None, row=1, col=c)
        return fig

    def create_confidence_grid(self, *, state: str):
        return self._grid(state=state, prediction=False)

    def create_prediction_grid(self, *, state: str):
        return self._grid(state=state, prediction=True)
    
    def create_combined_grid(self, *, state: str):
        """Create a grid with both confidence and prediction intervals."""
        x_grid = self._x_grid()
        fig = make_subplots(rows=3, cols=4, subplot_titles=self._MONTHS,
                            shared_xaxes=True, shared_yaxes=False, vertical_spacing=0.05)

        # Show legend only once per dataset / band type
        showleg_line = {ds: True for ds in self.datasets}
        showleg_ci = {ds: True for ds in self.datasets}
        showleg_pi = {ds: True for ds in self.datasets}
        showleg_era5 = True
        showleg_merra2 = True

        for idx, month in enumerate(self._MONTHS, start=1):
            r = (idx - 1)//4 + 1
            c = (idx - 1)%4 + 1
            
            # Add regression bands for each dataset
            for ds in self.datasets:
                ints, slps, sigma = self._subset_coeff(state=state, month=month, dataset=ds)
                
                # Center the x-grid values for the regression calculation
                x_centered = x_grid - self.dataset_means[ds]
                
                # Calculate confidence intervals (CI)
                ci_lo, ci_hi = self._bootstrap_band(x_centered, ints, slps,
                                                   resid_std=None, alpha=self.alpha)
                
                # Calculate prediction intervals (PI)
                pi_lo, pi_hi = self._bootstrap_band(x_centered, ints, slps,
                                                   resid_std=sigma, alpha=self.alpha)
                
                y_hat = np.median(ints) + np.median(slps) * x_centered

                # Plot median line
                fig.add_trace(go.Scatter(x=x_grid, y=y_hat, mode="lines",
                                          line=dict(color=self._LINE_COLOUR[ds], width=2),
                                          name=f"{ds} median" if showleg_line[ds] else None,
                                          legendgroup=f"{ds}_median",
                                          showlegend=showleg_line[ds]),
                               row=r, col=c)
                
                # Plot prediction interval (wider, lighter)
                fig.add_trace(go.Scatter(x=np.concatenate([x_grid, x_grid[::-1]]),
                                          y=np.concatenate([pi_lo, pi_hi[::-1]]), fill="toself",
                                          fillcolor=self._BAND_COLOUR[ds].replace("0.3)", "0.15)"),  # Make lighter
                                          line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
                                          name=f"{ds} PI" if showleg_pi[ds] else None,
                                          legendgroup=f"{ds}_pi",
                                          showlegend=showleg_pi[ds]),
                               row=r, col=c)
                
                # Plot confidence interval (narrower, darker)
                fig.add_trace(go.Scatter(x=np.concatenate([x_grid, x_grid[::-1]]),
                                          y=np.concatenate([ci_lo, ci_hi[::-1]]), fill="toself",
                                          fillcolor=self._BAND_COLOUR[ds],
                                          line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
                                          name=f"{ds} CI" if showleg_ci[ds] else None,
                                          legendgroup=f"{ds}_ci",
                                          showlegend=showleg_ci[ds]),
                               row=r, col=c)
                
                showleg_line[ds] = False
                showleg_ci[ds] = False
                showleg_pi[ds] = False

            # Always add historical data for both datasets
            for dataset_name in ["era5", "merra2"]:
                dataset_data = self.hist_df[(self.hist_df["Dataset"].str.lower() == dataset_name) & 
                                           (self.hist_df["Region"] == state) & 
                                           (self.hist_df["Month"] == month)]
                
                if not dataset_data.empty:
                    # Use different colors for each dataset
                    colors = {"era5": "#5D1D95", "merra2": "#5DA9E9"}
                    show_legend = showleg_era5 if dataset_name == "era5" else showleg_merra2
                    fig.add_trace(go.Scatter(
                        x=dataset_data["Global_Temp"], 
                        y=dataset_data["Average_Temperature"],
                        mode="markers", 
                        name=f"{dataset_name.upper()} Historical" if show_legend else None,
                        legendgroup=f"{dataset_name.upper()}_historical",
                        marker=dict(color=colors[dataset_name], size=4),
                        showlegend=show_legend
                    ), row=r, col=c)
                    
                    if dataset_name == "era5":
                        showleg_era5 = False
                    else:
                        showleg_merra2 = False

        # Update layout with consistent styling
        title = f"{state}: 95% Confidence & Prediction Intervals with Historical Data"
        fig.update_layout(
            template="simple_white", 
            height=800,  
            width=1200,   
            title=dict(text=title, x=0.5, font=dict(size=14)),
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.01,  # Closer to plot to prevent cutoff
                font=dict(size=12)
            ),
            margin=dict(l=60, r=120, t=60, b=60),  # Add margins to prevent cutoff
            autosize=True  # Allow plot to resize to container
        )
        
        # Add axis labels - centered for the entire plot area
        fig.add_annotation(
            text="Global Temperature (°C)",
            xref="paper", yref="paper",
            x=0.5, y=-0.08,  # Center of plot area, below subplots
            showarrow=False,
            font=dict(size=14, color="black"),
            xanchor="center"
        )
        
        fig.add_annotation(
            text="Regional Temperature (°C)",
            xref="paper", yref="paper",
            x=-0.04, y=0.5,  # Left of plot area, middle height
            showarrow=False,
            font=dict(size=14, color="black"),
            textangle=-90,  # Rotate 90 degrees
            xanchor="center", yanchor="middle"
        )
        
        return fig

class Plots:
    def __init__(self, df):
        """Figure 1: Base figure generated with plot_slope_map(). Significant trends identified with significant_trends_by_month(), and significance markers
         filled in using Adobe Illustrator.
         """
        self.df = df
        self.data_df = pd.read_csv(r"full_processed_data.csv")
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        # order df by month
        self.df = self.df.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))
        self.era5_global_mean_temp = self.data_df[(self.data_df["Dataset"] == "era5") & (self.data_df["Month"] == "Jan")]["Global_Temp"].mean()
        self.merra2_global_mean_temp = self.data_df[(self.data_df["Dataset"] == "merra2") & (self.data_df["Month"] == "Jan")]["Global_Temp"].mean()

    def plot_slope_map(self):
        """
        Plot a heatmap of the slope values for each region and month.
        """
        # create a choropleth map of the trends
        zero_point = abs(min(self.df["Pooled_Slope"]))/(max(self.df["Pooled_Slope"]) - min(self.df["Pooled_Slope"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]
        fig = px.choropleth(
            self.df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Pooled_Slope",
            title = "Trends by Month",
            scope = "usa",
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0,
            facet_row_spacing = 0,
            color_continuous_scale = color_scale
        )
        fig.update_layout(width = 1500, height = 1000)

        return fig
    
    def significant_trends_by_month(self, month):
        """
        Plot a heatmap of the slope values for each region and month.
        """
        month_df = self.df[self.df["Month"] == month]
        print(month_df[month_df["Slope_p"] < 0.05])

    def example_regression(self, dataset):
        example_state = "ND"
        all_data = pd.read_csv(r"full_processed_data.csv")
        global_temps = all_data["Global_Temp"].values
        min_global_temp, max_global_temp = min(global_temps), max(global_temps)
        x_range = np.linspace(min_global_temp, max_global_temp, 100)
        region_data = all_data[(all_data["Region"] == example_state) & (all_data["Dataset"] == dataset)]
        print(region_data)

        # historical data
        fig = px.scatter(region_data, x = "Global_Temp", y = "Average_Temperature", facet_col = "Month", facet_col_wrap = 4)
        
        # Add regression lines for each month
        months = region_data["Month"].unique()
        for i, month in enumerate(months):
            month_data = region_data[region_data["Month"] == month]
            
            if len(month_data) > 1:  # Need at least 2 points for regression
                # Fit linear regression
                x = month_data["Global_Temp"].values
                y = month_data["Average_Temperature"].values
                
                # Simple linear regression
                coeffs = np.polyfit(x, y, 1)
                slope, intercept = coeffs
                
                # Create regression line points
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                
                # Add regression line to the appropriate subplot
                # Calculate subplot position
                row = i // 4 + 1
                col = i % 4 + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="red", width=2),
                        name=f"{month} Regression",
                        showlegend=False
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(width = 1500, height = 1000)
        # Make y-axes independent for each facet
        fig.update_yaxes(showticklabels=True, matches=None)
        
        return fig
    
    def example_regression_with_pooled_results(self, dataset):
        """
        Alternative version that uses pre-computed regression results from the pooled analysis.
        """
        example_state = "ND"
        global_temp_mean = self.era5_global_mean_temp if dataset == "era5" else self.merra2_global_mean_temp
        
        # Load pooled regression results
        pooled_results = pd.read_csv(r"Regression Results/pooled_bootstrap_results.csv")
        state_results = pooled_results[pooled_results["Region"] == example_state]
        
        region_data = self.data_df[(self.data_df["Region"] == example_state) & (self.data_df["Dataset"] == dataset)]
        
        # Make plots
        fig = make_subplots(rows = 3, cols = 4)
        months = region_data["Month"].unique()
        for i, month in enumerate(months):

            month_data = region_data[region_data["Month"] == month]
            month_results = state_results[state_results["Month"] == month]
            
            if not month_results.empty and len(month_data) > 0:
                # Get regression coefficients from pooled results
                intercept = month_results["Pooled_Intercept"].iloc[0]
                slope = month_results["Pooled_Slope"].iloc[0]
                
                # Create regression line
                x_min, x_max = month_data["Global_Temp"].min(), month_data["Global_Temp"].max()
                x_line = np.linspace(x_min, x_max, 100)
                y_line = intercept + slope * (x_line - x_line.mean())
                
                # Add regression line to subplot
                row = i // 4 + 1
                col = i % 4 + 1
                
                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        line=dict(color="#56A3A6", width=2),
                        name=f"{month} Pooled Regression",
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # historical data
                fig.add_trace(go.Scatter(x = month_data["Global_Temp"], y = month_data["Average_Temperature"], mode = "markers", name = "Historical Data", showlegend = False, marker = dict(color = "#CBC3E3")), row = row, col = col)
        
        fig.update_layout(width=1500, height=1000)
        fig.update_yaxes(showticklabels=True, matches=None)
        
        return fig

    def plot_clustering_results(self, results_file_path = "Clustering Results/clustering_results_pooled_bootstrap_data_6.csv"):
        cluster_results = pd.read_csv(results_file_path)
        cluster_results = cluster_results.sort_values(by = "Cluster")
        cluster_results["Cluster"] = cluster_results["Cluster"].astype(str)
        cluster_results["Cluster"] = cluster_results["Cluster"].replace({"0": "1", "1": "2", "2": "3", "3": "4", "4": "5", "5": "6"})

        color_map = {"1": "#ff8389", "2": "#ee5396", "3": "#be95ff", "4": "#24a148", "5": "#33b1ff", "6": "#007d79"}
        fig = px.choropleth(cluster_results, 
                            locations = "Region", 
                            locationmode = "USA-states", 
                            color = "Cluster", 
                            title = "Clustering Results", 
                            scope = "usa",
                            color_discrete_map = color_map)
        fig.update_layout(width = 1200, height = 700)
        fig.update_coloraxes(colorbar_title = "Cluster")

        return fig

    def plot_cluster_trends(self):
        cluster_results = pd.read_csv(r"Clustering Results/clustering_results_pooled_bootstrap_data_6.csv")
        cluster_results = cluster_results.sort_values(by = "Cluster")
        cluster_results["Cluster"] = cluster_results["Cluster"].astype(str)
        cluster_results["Cluster"] = cluster_results["Cluster"].replace({"0": "1", "1": "2", "2": "3", "3": "4", "4": "5", "5": "6"})
        
        # Add cluster information to self.df by matching Region columns
        # Create a new DataFrame in memory with the cluster column added
        df_with_clusters = self.df.copy()
        
        # Merge the cluster information based on Region
        df_with_clusters = df_with_clusters.merge(
            cluster_results[['Region', 'Cluster']], 
            on='Region', 
            how='left'
        )
        
        # Ensure Cluster is numeric for proper sorting
        df_with_clusters['Cluster'] = pd.to_numeric(df_with_clusters['Cluster'])
        
        # Sort by Cluster first, then by Month in calendar order
        df_with_clusters = df_with_clusters.sort_values(by=['Cluster', 'Month'], 
                                                       key=lambda x: x.map(lambda y: self.months.index(y)) if x.name == 'Month' else x)

        # Create categorical ordering for Month to ensure sequential display
        df_with_clusters['Month'] = pd.Categorical(df_with_clusters['Month'], 
                                                  categories=self.months, 
                                                  ordered=True)

        # Create box plot manually to control colors properly
        fig = make_subplots(rows=3, cols=4, subplot_titles=self.months, vertical_spacing=0.08)
        
        # Calculate overall y-axis range for consistent scaling
        y_min = df_with_clusters['Pooled_Slope'].min()
        y_max = df_with_clusters['Pooled_Slope'].max()
        
        color_map = {"1": "#ff8389", "2": "#ee5396", "3": "#be95ff", "4": "#24a148", "5": "#33b1ff", "6": "#007d79"}
        
        # Add box plots for each month and cluster
        for month_idx, month in enumerate(self.months):
            row = month_idx // 4 + 1
            col = month_idx % 4 + 1
            
            for cluster in sorted(df_with_clusters['Cluster'].unique()):
                cluster_data = df_with_clusters[(df_with_clusters['Month'] == month) & 
                                              (df_with_clusters['Cluster'] == cluster)]
                
                if not cluster_data.empty:
                    fig.add_trace(
                        go.Box(
                            x=[cluster] * len(cluster_data),
                            y=cluster_data['Pooled_Slope'],
                            name=f"Cluster {cluster}",
                            marker_color=color_map[str(cluster)],
                            showlegend=False if month_idx > 0 else True,
                            legendgroup=f"Cluster {cluster}"
                        ),
                        row=row, col=col
                    )

        # Calculate the number of regions in each cluster
        cluster_counts = df_with_clusters.groupby('Cluster')['Region'].nunique()
        
        # Update the legend to include cluster member counts
        for trace in fig.data:
            if 'Cluster' in trace.name:
                cluster_num = trace.name.split()[-1]  # Extract cluster number
                count = cluster_counts.get(int(cluster_num), 0)
                trace.name = f"Cluster {cluster_num} ({count})"
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title="Trends by Cluster"
        )
        
        # Update x-axes to show all cluster numbers and add labels
        for i in range(1, 13):  # 12 subplots
            row = (i - 1) // 4 + 1
            col = (i - 1) % 4 + 1
            
            # Show all cluster numbers on x-axis
            fig.update_xaxes(
                tickmode='array',
                tickvals=[1, 2, 3, 4, 5, 6],
                ticktext=['1', '2', '3', '4', '5', '6'],
                row=row, col=col
            )
            
            # Add y-axis label only for leftmost column, 2nd row
            if col == 1 and row == 2:  # Leftmost column, 2nd row only
                fig.update_yaxes(title_text="Pooled Slope", row=row, col=col, range=[y_min, y_max])
            else:
                fig.update_yaxes(title_text="", row=row, col=col, range=[y_min, y_max])
        
        fig.add_annotation(
            x = 0.5,
            y = -0.075,
            text = "Cluster",
            showarrow = False,
            xref = "paper",
            yref = "paper",
            font = dict(size = 14)
        )
        
        return fig

    def plot_highest_lowest_trends(self):
        df = self.df
        df = df.groupby("Region")["Pooled_Slope"].mean().reset_index()
        top_5 = df.nlargest(5, "Pooled_Slope")
        bottom_5 = df.nsmallest(5, "Pooled_Slope")

        df = pd.concat([top_5, bottom_5])
        df["color"] = ["highest" for i in range(len(df))]
        df["color"][5:] = ["lowest" for i in range(5)]
        color_map = {"highest": "maroon", "lowest": "#053061"}
        fig = px.bar(df, 
                     x = "Region", 
                     y = "Pooled_Slope", 
                     color = "color", 
                     color_discrete_map = color_map,
                     title = "States with Weakest and Strongest Trends")
        fig.update_layout(height = 500, width = 750, showlegend = False)

        return fig
    
    def plot_monthly_distributions(self):
        df = self.df
        max_slope = df["Pooled_Slope"].max()
        min_slope = df["Pooled_Slope"].min()
        color_scale = n_colors(hex_to_rgb("#80ddff"), hex_to_rgb("#bb80ff"), 12, colortype = "tuple")
        color_scale = ["rgb" + str(color) for color in color_scale]

        fig = make_subplots(rows = 4, cols = 3, 
                            subplot_titles = [f"{month}" for month in self.months])
        for i, month in enumerate(self.months):
            distribution = df[df["Month"] == month]
            fig.add_trace(go.Histogram(x = distribution["Pooled_Slope"], name = f"{month}", marker_color = color_scale[i]),  row = i // 3 + 1, col = i % 3 + 1)
            fig.update_xaxes(range = [min_slope, max_slope], row = i // 3 + 1, col = i % 3 + 1)
            fig.update_yaxes(range = [0, 20], row = i // 3 + 1, col = i % 3 + 1)
            fig.add_vline(x = distribution["Pooled_Slope"].median(), row = i // 3 + 1, col = i % 3 + 1, line = dict(color = "orange"))
            fig.add_annotation(x = distribution["Pooled_Slope"].median() + 0.75, y = 18, text = f"{distribution['Pooled_Slope'].median():.2f}", 
                               showarrow = False, font = dict(color = "orange"), row = i // 3 + 1, col = i % 3 + 1)
        
        fig.update_layout(title = f"Slope Distribution by Month", height = 750, width = 850, showlegend = False)
        fig.update_xaxes(title_text = "Pooled_Slope", row = 4, col = 2)
        fig.add_annotation(x = -0.075, 
                           y = 0.5, 
                           text = "Frequency", 
                           showarrow = False, 
                           xref = "paper", 
                           yref = "paper", 
                           font = dict(size = 14),
                           textangle = -90)

        return fig

class RiskAssessment:
    def __init__(self, state = "MA"):
        self.coef_df = pd.read_csv("Regression Results/pooled_bootstrap_results_t2mmax.csv")
        self.state = state
        self.full_state_name = self.abbreviation_to_full_name()[state]
        self.state_populations = pd.read_csv(r"state_populations.csv")
        self.state_flowers = pd.read_csv(r"state_flowers.csv")

    def abbreviation_to_full_name(self):
        df = pd.read_csv(r"state_cmi.csv")
        abbreviation_dict = {key: value for value, key in zip(df["state"], df["abbreviation"])}

        return abbreviation_dict

    def get_risk_assessment(self):
        lower_third_percentile = np.percentile(self.coef_df.groupby("Region")["Pooled_Slope"].mean(), 33.33)
        upper_third_percentile = np.percentile(self.coef_df.groupby("Region")["Pooled_Slope"].mean(), 66.66)
        coef = self.coef_df[self.coef_df["Region"] == self.state]["Pooled_Slope"].values.mean()

        if coef > upper_third_percentile:
            return "HIGH", "red"
        elif lower_third_percentile <= coef <= upper_third_percentile:
            return "MEDIUM", "orange"
        else:
            return "LOW", "green"

    def risk_assessment_div_element(self, n_clicks):
        from dash import html
        from dash_mantine_components import Text

        risk, color = self.get_risk_assessment()

        div_element = html.Div(
            children = [Text(f"{self.full_state_name}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 30, "color": "black"}, id = f"state-name-{n_clicks}"),
                        Text(f"Population: {self.state_populations[self.state_populations['State'] == self.full_state_name]['Population'].values[0]} ｜ State flower: {self.state_flowers[self.state_flowers['State'] == self.full_state_name]['Common name'].values[0]}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 20, "color": "black"}, id = f"state-info-{n_clicks}"),
                        html.Div(children = [Text(f"Warming Risk:", style = {"fontSize": 20, "color": "black"}, className = "animate__animated animate__fadeInRightBig animate__slow", id = f"risk-label-{n_clicks}"), Text(f"{risk}", className = "animate__animated animate__fadeInRightBig animate__slow", style = {"fontSize": 20, "color": color, "marginLeft": "10px"}, id = f"risk-value-{n_clicks}")], style = {"display": "flex", "alignItems": "left"})]
        )
        return div_element

class KMeansClustering:
    def __init__(self):
        self.df = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def generate_clusters(self, csv_export_basename, n_clusters = 6):
        from sklearn.cluster import KMeans

        # Prepare data
        df = self.df.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))
        values = "Pooled_Slope"
        clustering_data = df.pivot(index = "Region", columns = "Month", values = values).reindex(columns = self.months)
        clustering_data_state_names = clustering_data.index.values

        # K-means clustering
        kmeans = KMeans(n_clusters = n_clusters, n_init = 10)
        kmeans.fit(clustering_data)
        results_df = pd.DataFrame({"Region": clustering_data_state_names, "Cluster": kmeans.labels_})
        results_df.to_csv(f"Clustering Results/{csv_export_basename}_{n_clusters}.csv", index = False)

class MannKendallTrendTest:
    def __init__(self):
        self.df = pd.read_csv("full_processed_data.csv")
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def average_datasets(self):
        # For each year, month, and region grouping, average the values of Average_Temperature 
        # between the era5 and merra2 datasets.
        # Group by Year, Month, Region and average the datasets
        averaged_data = self.df.groupby(['Year', 'Month', 'Region'])['Average_Temperature'].mean().reset_index()
        
        # Sort by Month in calendar order
        averaged_data['Month'] = pd.Categorical(averaged_data['Month'], categories=self.months, ordered=True)
        averaged_data = averaged_data.sort_values(['Region', 'Month', 'Year'])
        
        return averaged_data

    def perform_mann_kendall_analysis(self):
        """
        For each month and region pair, conduct a Mann-Kendall analysis using mk.original_test.
        """
        averaged_data = self.average_datasets()
        results = []
        
        # Get unique regions
        regions = averaged_data['Region'].unique()
        
        for month in self.months:
            for region in regions:
                # Get data for this month and region
                subset = averaged_data[(averaged_data['Month'] == month) & 
                                     (averaged_data['Region'] == region)]

                if len(subset) >= 3:  # Need at least 3 points for Mann-Kendall test
                    # Sort by year to ensure chronological order
                    subset = subset.sort_values('Year')
                    
                    # Perform Mann-Kendall test
                    try:
                        mk_result = mk.original_test(subset['Average_Temperature'].values)
                        
                        results.append({
                            'Month': month,
                            'Region': region,
                            'trend': mk_result.trend,
                            'h': mk_result.h,
                            'p': mk_result.p,
                            'z': mk_result.z,
                            'Tau': mk_result.Tau,
                            's': mk_result.s,
                            'var_s': mk_result.var_s,
                            'slope': mk_result.slope,
                            'intercept': mk_result.intercept
                        })
                    except Exception as e:
                        print(f"Error processing {month}/{region}: {e}")
                        # Add row with NaN values for failed tests
                        results.append({
                            'Month': month,
                            'Region': region,
                            'trend': 'no trend',
                            'h': False,
                            'p': np.nan,
                            'z': np.nan,
                            'Tau': np.nan,
                            's': np.nan,
                            'var_s': np.nan,
                            'slope': np.nan,
                            'intercept': np.nan
                        })
                else:
                    print(f"Insufficient data for {month}/{region}: {len(subset)} observations")
                    # Add row with NaN values for insufficient data
                    results.append({
                        'Month': month,
                        'Region': region,
                        'trend': 'no trend',
                        'h': False,
                        'p': np.nan,
                        'z': np.nan,
                        'Tau': np.nan,
                        's': np.nan,
                        'var_s': np.nan,
                        'slope': np.nan,
                        'intercept': np.nan
                    })
        
        return pd.DataFrame(results)

    def plot_mann_kendall_results(self, results_df):
        """
        Plot the Mann-Kendall results on a choropleth map, faceted by month.
        Shows the magnitude of the trend using the Theil-Sen estimator (slope).
        """
        # Filter out rows with NaN slopes
        plot_df = results_df.dropna(subset=['slope'])
        
        # Create choropleth map using the slope (Theil-Sen estimator)
        zero_point = abs(min(plot_df["slope"]))/(max(plot_df["slope"]) - min(plot_df["slope"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]
        fig = px.choropleth(
            plot_df,
            locations="Region",
            locationmode="USA-states",
            color="slope",
            facet_col="Month",
            facet_col_wrap=4,
            scope="usa",
            title="Mann-Kendall Test Results",
            color_continuous_scale=color_scale,
            range_color=[plot_df['slope'].min(), plot_df['slope'].max()]
        )
        
        fig.update_layout(
            width=1500,
            height=1000
        )
        
        # Update colorbar title
        fig.update_coloraxes(colorbar_title="Temperature Trend (°C/year)")
        
        return fig

    def run_complete_analysis(self):
        """
        Run the complete Mann-Kendall analysis and return both results and plot.
        """
        results = self.perform_mann_kendall_analysis()
        fig = self.plot_mann_kendall_results(results)
        
        return fig

if __name__ == "__main__":
    # Temperature-temperature plot
    # fig = PlotlySlopeMap(coeff_df = pd.read_csv("Regression Results/uncertainty_intervals_with_prediction_bands.csv"), 
    #                     hist_df = pd.read_csv("full_processed_data.csv")).create_confidence_grid(state = "MA")
    # fig.show()

    # individual state regression
    # df = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")
    # fig = Plots(df).example_regression_with_pooled_results("merra2")
    # fig.show()

    # chloropleth slope
    # df = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")
    # fig = Plots(df).plot_slope_map()
    # fig.show()

    # clustering analysis
    # KMeansClustering().generate_clusters("clustering_results_pooled_bootstrap_data", n_clusters = 4)

    # clustering results
    # df = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")
    # fig = Plots(df).plot_clustering_results(results_file_path = "Clustering Results/clustering_results_pooled_bootstrap_data_5.csv")
    # fig.show()

    # cluster trends
    # df = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")
    # fig = Plots(df).plot_cluster_trends()
    # fig.write_image("Clustering Results/cluster_trends_6.svg")

    # highest and lowest trends
    # df = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")
    # fig = Plots(df).plot_highes_lowest_trends()
    # fig.write_image("Publication Plots/highest_lowest_trends.svg")

    # monthly distributions
    # df = pd.read_csv("Regression Results/pooled_bootstrap_results.csv")
    # fig = Plots(df).plot_monthly_distributions()
    # fig.show()

    # Mann-Kendall trend analysis
    # mk_analyzer = MannKendallTrendTest()
    # fig = mk_analyzer.run_complete_analysis()
    # fig.write_image("Publication Plots/mann_kendall_trend_analysis.svg")

    # ==== APP FUNCTIONS ====
    # fig = AppFunctionsforPooledData(scenario = "aa").make_by_year_plot(region = "MA")
    # fig.show()
    
    # PRE-GENERATE ALL PLOTS FOR WEBAPP (run once)
    # Uncomment to generate all static HTML files:
    # app = AppFunctionsforPooledData(scenario="aa")  # scenario doesn't matter for pre-generation
    # metadata = app.pregenerate_all_plots(output_dir="webapp_plots")
    # print(f"Generated {metadata['total_plots']} plots in {metadata['generated_at']}")

    # workflow
    """
    ERA5 data downloaded from download_data function in ERA5Data class in data_retrieval.py
    ERA5 data converted to json using make_results_dict function in ERA5 in archived_analysis.py (includes rescaling)
    MERRA2 data downloaded from RetrieveSingleVariable class in data_retrieval.py
    MERRA2 data converted to json using aggregate_inside_points_temp_data function in CollectRegionalData class in data_retrieval.py
    CompareRegressionResults class in archived_analysis.py creates the combined csv (need to update)
    For generated static plots, there's a problem with how the uncertainty intervals are generated, which needs to be fixed
    """
    # df = pd.read_csv(r"full_processed_data_t2mmin.csv")
    # PooledEstimator(df = df).save_pooled_bootstrap(outfile = "Regression Results/pooled_bootstrap_results_t2mmin.csv")

    # Webapp plot SVGs for publication
    # fig = AppFunctionsforPooledData(scenario = "aa").make_by_year_plot(region = "MA")
    # fig.show()
    fig = AppFunctionsforPooledData(scenario = "ct", var = "T2MMIN").make_by_temp_plot(region = "MO")
    fig.write_image("MO_min_temp_example.svg")

    # R2 validation
    # fig = AppFunctionsforPooledData(scenario = "aa").plot_r2()
    # fig.write_image("Publication Plots/r2_validation.svg")
