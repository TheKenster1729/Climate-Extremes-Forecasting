import pandas as pd
import numpy as np
import os
import plotly.graph_objects as go
import netCDF4 as nc
import json
import plotly.express as px
from scipy.stats import norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

class AppFunctionsforPooledData:
    def __init__(self, scenario, var = "T2MMAX", end_year = 2050):
        self.scenario = scenario
        self.var = var
        self.end_year = end_year
        self.data = pd.read_csv(r"full_processed_data.csv")
        self.regression_years = self.data.Year.unique()
        # global data is the same for all months, so we can use any month to get the global mean temp
        self.era5_global_mean_temp = self.data[(self.data["Year"].isin(self.regression_years)) & (self.data["Dataset"] == "era5") & (self.data["Month"] == "Jan")]["Global_Temp"].mean()
        self.merra2_global_mean_temp = self.data[(self.data["Year"].isin(self.regression_years)) & (self.data["Dataset"] == "merra2") & (self.data["Month"] == "Jan")]["Global_Temp"].mean()

        self.regression_results = pd.read_csv(r"Regression Results/pooled_bootstrap_results.csv")
        self.uncertainty_intervals = pd.read_csv(r"Regression Results/uncertainty_intervals.csv")

    def get_merra2_historical_data(self, region):
        m2_data = self.data[(self.data["Dataset"] == "merra2") & (self.data["Region"] == region)]
        return m2_data

    def get_era5_historical_data(self, region):
        e5_data = self.data[(self.data["Dataset"] == "era5") & (self.data["Region"] == region)]
        return e5_data
    
    def make_by_temp_plot(self, region):
        # historical data
        historical_df = self.data[self.data["Region"] == region]
        historical_df["Dataset"] = historical_df["Dataset"].apply(str.upper)

        # Sort historical data by month in calendar order
        months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        historical_df['Month'] = pd.Categorical(historical_df['Month'], categories=months, ordered=True)
        historical_df = historical_df.sort_values('Month').reset_index(drop=True)

        # Load pooled bootstrap results
        region_uncertainty_intervals = self.uncertainty_intervals[self.uncertainty_intervals["Region"] == region]

        # plot historical data
        fig = make_subplots(rows=3, cols=4, subplot_titles=months)
        # Add regression lines and uncertainty intervals for each month  
        for i, month in enumerate(months):
            month_data = historical_df[historical_df["Month"] == month]
            
            # Get uncertainty intervals for this month
            month_intervals = region_uncertainty_intervals[region_uncertainty_intervals["Month"] == month]
            
            if not month_intervals.empty:
                # Get x range for smooth regression lines
                x_min = month_data["Global_Temp"].min()
                x_max = month_data["Global_Temp"].max()
                x_range = np.linspace(x_min, x_max, 100)
                
                # ERA5 regression lines
                era5_x_centered = x_range - self.era5_global_mean_temp
                era5_y_median = month_intervals["Intercept_50th"].iloc[0] + month_intervals["Slope_50th"].iloc[0] * era5_x_centered
                era5_y_upper = month_intervals["Intercept_95th"].iloc[0] + month_intervals["Slope_95th"].iloc[0] * era5_x_centered
                era5_y_lower = month_intervals["Intercept_5th"].iloc[0] + month_intervals["Slope_5th"].iloc[0] * era5_x_centered
                
                # MERRA2 regression lines  
                merra2_x_centered = x_range - self.merra2_global_mean_temp
                merra2_y_median = month_intervals["Intercept_50th"].iloc[0] + month_intervals["Slope_50th"].iloc[0] * merra2_x_centered
                merra2_y_upper = month_intervals["Intercept_95th"].iloc[0] + month_intervals["Slope_95th"].iloc[0] * merra2_x_centered
                merra2_y_lower = month_intervals["Intercept_5th"].iloc[0] + month_intervals["Slope_5th"].iloc[0] * merra2_x_centered
                
                # Calculate subplot position (now matches plotly's forced order)
                row = i // 4 + 1
                col = i % 4 + 1

                # ERA5 historical data
                fig.add_trace(
                    go.Scatter(
                        x=month_data[month_data["Dataset"] == "ERA5"]["Global_Temp"],
                        y=month_data[month_data["Dataset"] == "ERA5"]["Average_Temperature"],
                        mode="markers",
                        marker=dict(color="#5D1D95"),
                        name="ERA5 Historical Data",
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # MERRA2 historical data
                fig.add_trace(
                    go.Scatter(
                        x=month_data[month_data["Dataset"] == "MERRA2"]["Global_Temp"],
                        y=month_data[month_data["Dataset"] == "MERRA2"]["Average_Temperature"],
                        mode="markers",
                        marker=dict(color="#5DA9E9"),
                        name="MERRA2 Historical Data",
                        showlegend=False
                    ),
                    row=row, col=col
                )

                # Add ERA5 uncertainty band
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_range, x_range[::-1]]),
                        y=np.concatenate([era5_y_upper, era5_y_lower[::-1]]),
                        fill='toself',
                        fillcolor='rgba(93, 29, 149, 0.2)',  # Light purple for ERA5
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name="ERA5 Uncertainty"
                    ),
                    row=row, col=col
                )
                
                # Add MERRA2 uncertainty band
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([x_range, x_range[::-1]]),
                        y=np.concatenate([merra2_y_upper, merra2_y_lower[::-1]]),
                        fill='toself',
                        fillcolor='rgba(93, 169, 233, 0.2)',  # Light blue for MERRA2
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False,
                        name="MERRA2 Uncertainty"
                    ),
                    row=row, col=col
                )
                
                # Add ERA5 median regression line
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=era5_y_median,
                        mode='lines',
                        line=dict(color='#5D1D95', width=2),  # ERA5 color
                        showlegend=False,
                        name="ERA5 Regression"
                    ),
                    row=row, col=col
                )
                
                # Add MERRA2 median regression line
                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=merra2_y_median,
                        mode='lines',
                        line=dict(color='#5DA9E9', width=2),  # MERRA2 color
                        showlegend=False,
                        name="MERRA2 Regression"
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            height=800, width=1200,
            margin=dict(t=100, l=40, r=40, b=40),
        )
        fig.update_xaxes(matches=None)
        fig.update_yaxes(matches=None)
        fig.for_each_xaxis(lambda x: x.update(title_text=""))
        fig.for_each_yaxis(lambda y: y.update(title_text=""))


        return fig

    def make_by_year_plot(self, region):
        pass
        
    def make_plots(self):
        pass

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
        years = self.df["Year"].unique()
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
                mer  = mer .set_index("Year").loc[yrs]
                x = era5["Global_Temp_c"].values      # aligned predictor
                y_e = era5["Average_Temperature"].values
                y_m = mer ["Average_Temperature"].values
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
                
                # Calculate percentiles
                int_5th = np.percentile(ints, 5)
                int_50th = np.percentile(ints, 50)
                int_95th = np.percentile(ints, 95)
                
                slp_5th = np.percentile(slps, 5)
                slp_50th = np.percentile(slps, 50)
                slp_95th = np.percentile(slps, 95)
                
                uncertainty_rows.append(dict(
                    Region=reg,
                    Month=mon,
                    Slope_5th=slp_5th,
                    Slope_50th=slp_50th,
                    Slope_95th=slp_95th,
                    Intercept_5th=int_5th,
                    Intercept_50th=int_50th,
                    Intercept_95th=int_95th
                ))
        
        return (pd.DataFrame(uncertainty_rows)
                .sort_values(["Region", "Month"])
                .reset_index(drop=True))

    def get_uncertainty_intervals_for_region_month(self, region: str, month: str) -> dict:
        """
        Get uncertainty intervals for a specific region and month.
        Returns a dict with slope and intercept percentiles.
        """
        uncertainty_df = self._produce_uncertainty_intervals()
        row = uncertainty_df[(uncertainty_df['Region'] == region) & 
                           (uncertainty_df['Month'] == month)]
        
        if row.empty:
            return None
            
        return {
            'slope_5th': row['Slope_5th'].iloc[0],
            'slope_50th': row['Slope_50th'].iloc[0], 
            'slope_95th': row['Slope_95th'].iloc[0],
            'intercept_5th': row['Intercept_5th'].iloc[0],
            'intercept_50th': row['Intercept_50th'].iloc[0],
            'intercept_95th': row['Intercept_95th'].iloc[0]
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
        self.df = self.df.set_index(['Region', 'Month'])
    
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
            row = self.df.loc[(region, month)]
            return {
                'slope_5th': row['Slope_5th'],
                'slope_50th': row['Slope_50th'], 
                'slope_95th': row['Slope_95th'],
                'intercept_5th': row['Intercept_5th'],
                'intercept_50th': row['Intercept_50th'],
                'intercept_95th': row['Intercept_95th']
            }
        except KeyError:
            return None
    
    def get_all_regions(self) -> list:
        """Get list of all available regions."""
        return self.df.index.get_level_values('Region').unique().tolist()
    
    def get_all_months(self) -> list:
        """Get list of all available months."""
        return self.df.index.get_level_values('Month').unique().tolist()
    
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
            return region_data[['Slope_5th', 'Slope_50th', 'Slope_95th']]
        except KeyError:
            return pd.DataFrame()

class PlotlySlopeMap:
    """
    Interactive USA‑states choropleth of pooled slopes
    (ERA5 + MERRA‑2, 1980‑2022) with a month slider and
    '×' overlay on non‑significant states.

    Parameters
    ----------
    csv_path   : str | Path
        Output of PooledEstimator.save_pooled_bootstrap().
    sig_level  : float, default 0.05
        P‑value threshold used to flag non‑significant slopes.
    """

    # approximate geographic centroids (lat, lon) for 50 states + DC
    _centroids = {
        "AL": (32.7,  -86.7),  "AK": (64.5, -152.3), "AZ": (34.3, -111.7),
        "AR": (34.8,  -92.2),  "CA": (37.2, -119.5), "CO": (39.0, -105.7),
        "CT": (41.6,  -72.7),  "DE": (39.1,  -75.5), "FL": (28.6,  -82.4),
        "GA": (32.7,  -83.3),  "HI": (20.8, -156.3), "ID": (44.1, -114.7),
        "IL": (40.0,  -89.2),  "IN": (40.1,  -86.1), "IA": (42.1,  -93.5),
        "KS": (38.5,  -98.0),  "KY": (37.5,  -85.3), "LA": (31.2,  -92.3),
        "ME": (45.3,  -69.2),  "MD": (39.0,  -76.8), "MA": (42.4,  -71.4),
        "MI": (44.7,  -85.6),  "MN": (46.3,  -94.3), "MS": (32.7,  -89.7),
        "MO": (38.4,  -92.4),  "MT": (46.9, -110.4), "NE": (41.5,  -99.7),
        "NV": (39.0, -117.0),  "NH": (44.0,  -71.6), "NJ": (40.1,  -74.7),
        "NM": (34.4, -106.1),  "NY": (42.9,  -75.6), "NC": (35.6,  -79.9),
        "ND": (47.5, -100.4),  "OH": (40.3,  -82.8), "OK": (35.6,  -97.5),
        "OR": (43.9, -120.6),  "PA": (41.2,  -77.2), "RI": (41.6,  -71.6),
        "SC": (33.8,  -80.9),  "SD": (44.3, -100.2), "TN": (35.8,  -86.4),
        "TX": (31.5,  -99.4),  "UT": (39.4, -111.7), "VT": (44.1,  -72.7),
        "VA": (37.5,  -78.7),  "WA": (47.5, -120.5), "WV": (38.6,  -80.6),
        "WI": (44.7,  -89.7),  "WY": (43.0, -107.6), "DC": (38.9,  -77.0)
    }

    _month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    def __init__(self, csv_path, sig_level: float = 0.05):
        self.csv_path  = Path(csv_path)
        self.sig_level = sig_level
        self.data      = pd.read_csv(self.csv_path)
        # Convert 'Month' column to categorical with proper order
        self.data['Month'] = pd.Categorical(self.data['Month'], categories=self._month_order, ordered=True)

        # Sort by 'Month'
        self.data = self.data.sort_values('Month').reset_index(drop=True)

        # symmetric colour range
        self._cmax = self.data["Pooled_Slope"].abs().max()

    # -----------------------------------------------------------------
    # public renderer
    # -----------------------------------------------------------------
    def make_figure(self):
        """
        Returns
        -------
        fig : plotly.graph_objects.Figure
            Interactive choropleth with month slider.
        """
        # choropleth trace with animation frames handled by Plotly
        zero_point = abs(min(self.data["Pooled_Slope"]))/(max(self.data["Pooled_Slope"]) - min(self.data["Pooled_Slope"]))
        color_scale = [(0, "#053061"), (zero_point, "white"), (1, "maroon")]
        fig = px.choropleth(
            self.data,
            locations = "Region",
            locationmode = "USA-states",
            scope = "usa",
            color = "Pooled_Slope",
            color_continuous_scale = color_scale,
            facet_col = "Month",
            facet_col_wrap = 4,
            facet_col_spacing = 0
        )
        
        return fig
     
class RiskAssessment:
    def __init__(self, dataset = "MERRA2", var = "T2MMAX", state = "MA"):
        self.dataset = dataset
        self.var = var
        self.state = state
        self.abbreviation_dict = self.abbreviation_to_full_name()
        self.state_populations = pd.read_csv(r"state_populations.csv")
        self.state_flowers = pd.read_csv(r"state_flowers.csv")
        self.full_state_name = self.abbreviation_dict[self.state]

    def load_data(self):
        if self.dataset == "MERRA2":
            data = json.load(open(r"MERRA2/JSON Files/Regional Aggregates/us-states-regions.json"))
        elif self.dataset == "ERA5":
            data = json.load(open(r"ERA5/Temperature Data/JSON Files/us-states-era5-t2m.json"))

        return data
    
    def get_global_average_temp(self):
        if self.dataset == "MERRA2":
            data = pd.read_csv(r"global_average_temp_by_year.csv")["Average"].values
        elif self.dataset == "ERA5":
            data = RegressionAnalysisComplete(dataset = "ERA5", var = self.var).get_X()

        return data

    def get_regression_results(self):
        if self.dataset == "MERRA2":
            data = pd.read_csv(r"Regression Results/MERRA2/Max Temp/regression_results-merra2.csv")
        elif self.dataset == "ERA5":
            data = pd.read_csv(r"Regression Results/ERA5/Max Temp/regression_results-era5.csv")

        return data

    def abbreviation_to_full_name(self):
        df = pd.read_csv(r"state_cmi.csv")
        abbreviation_dict = {key: value for value, key in zip(df["state"], df["abbreviation"])}

        return abbreviation_dict

    def get_risk_assessment(self):
        regression_data = self.get_regression_results()
        lower_third_percentile = np.percentile(regression_data.groupby("Region")["Slope"].mean(), 33.33)
        upper_third_percentile = np.percentile(regression_data.groupby("Region")["Slope"].mean(), 66.66)
        coef = regression_data[regression_data["Region"] == self.state]["Slope"].values.mean()

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

        # div_element = html.Div(
        #     children = [html.H2(self.full_state_name),
        #                 html.H4(f"Population: {self.state_populations[self.state_populations['State'] == self.full_state_name]['Population'].values[0]}｜State flower: {self.state_flowers[self.state_flowers['State'] == self.full_state_name]['Common name'].values[0]}"),
        #                 html.Div(children = [html.H4(children = f"Warming Risk: ", style = {"marginRight": "10px"}), html.H4(children = f"{risk}", style = {"color": color, "display": "inline"})], style = {"display": "flex", "alignItems": "left"}),
        #                 ]
        # )
        return div_element

class PatternFinding:
    def __init__(self, dataset = "MERRA2", var = "T2MMAX", merra_2_timeframe = False):
        self.dataset = dataset
        self.var = var
        self.merra_2_timeframe = merra_2_timeframe
        self.data = self.load_data()
        self.months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    def load_data(self):
        if self.dataset == "MERRA2":
            df = pd.read_csv(r"Regression Results/MERRA2/regression_results-MERRA2-T2MMAX.csv")
        elif self.dataset == "ERA5":
            if self.merra_2_timeframe:
                df = pd.read_csv(r"Regression Results/ERA5/regression_results-ERA5-T2MMAX-merra2_timeframe.csv")
            else:
                df = pd.read_csv(r"Regression Results/ERA5/regression_results-ERA5-T2MMAX.csv")
        elif self.dataset == "combined":
            df = pd.read_csv(r"Regression Results/combined_slopes-T2MMAX-merra2_timeframe.csv")

        return df
    
    def preprocess_data_for_clustering(self):
        df = self.load_data()

        # order by month
        df = df.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))

        # represent each state by monthly trend vector
        values = "Slope in Original Units" if self.dataset != "combined" else "Combined Slope"
        clustering_data = df.pivot(index='Region', columns='Month', values=values).reindex(columns = self.months)
        clustering_data_state_names = clustering_data.index.values

        return clustering_data.values, clustering_data_state_names
    
    def k_means_clustering(self, n_clusters = 6):
        from sklearn.cluster import KMeans

        clustering_data, clustering_data_state_names = self.preprocess_data_for_clustering()

        kmeans = KMeans(n_clusters = n_clusters, n_init = 10)
        kmeans.fit(clustering_data)
        results_df = pd.DataFrame({"Region": clustering_data_state_names, "Cluster": kmeans.labels_})
        results_df.to_csv(f"clustering_results_{self.dataset}_{self.var}_{kmeans.n_clusters}.csv", index = False)
    
    def plot_clustering_results(self, n_clusters = 6):
        if os.path.exists(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv"):
            results_df = pd.read_csv(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv")
        else:
            self.k_means_clustering(n_clusters = n_clusters)
            results_df = pd.read_csv(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv")
        
        results_df.sort_values(by = ["Cluster"], inplace = True)
        results_df["Cluster"] = results_df["Cluster"] + 1
        results_df["Cluster"] = results_df["Cluster"].astype(str)

        color_map = {"1": "#80ddff", "2": "#bb80ff", "3": "#ffee80", "4": "#4d8599", "5": "#ddff80", "6": "#ffa280"}
        fig = px.choropleth(
            data_frame = results_df,
            locations = "Region",
            locationmode = "USA-states",
            color = "Cluster",
            scope = "usa",
            color_discrete_map = color_map,
            height = 600,
            width = 1000
        )
        fig.update_layout(title = "K-Means Clustering Results")

        return fig

    def plot_highest_lowest_states(self):
        df = self.load_data().groupby("Region")["Combined Slope"].mean().reset_index()
        top_5 = df.nlargest(5, 'Combined Slope')
        bottom_5 = df.nsmallest(5, 'Combined Slope')

        # Combine them into one DataFrame
        result = pd.concat([top_5, bottom_5])
        result["color"] = ["highest" for i in range(len(result))]
        result["color"][5:] = ["lowest" for i in range(5)]
        color_map = {"highest": "maroon", "lowest": "#053061"}
        print(result)

        fig = px.bar(
            data_frame = result,
            x = "Region",
            y = "Combined Slope",
            title = "States with Weakest and Strongest Trends",
            color = "color",
            color_discrete_map = color_map
        )
        fig.update_layout(height = 500, width = 750, showlegend = False)

        return fig

    def plot_monthly_distributions(self):
        from plotly.subplots import make_subplots
        from plotly.colors import n_colors, hex_to_rgb

        df = self.load_data()
        max_slope = df["Combined Slope"].max()
        min_slope = df["Combined Slope"].min()
        color_scale = n_colors(hex_to_rgb("#80ddff"), hex_to_rgb("#bb80ff"), 12, colortype = "tuple")
        color_scale = ["rgb" + str(color) for color in color_scale]

        fig = make_subplots(rows = 4, cols = 3, 
                            subplot_titles = [f"{month}" for month in self.months])
        for i, month in enumerate(self.months):
            distribution = df[df["Month"] == month]
            fig.add_trace(go.Histogram(x = distribution["Combined Slope"], name = f"{month}", marker_color = color_scale[i]),  row = i // 3 + 1, col = i % 3 + 1)
            fig.update_xaxes(range = [min_slope, max_slope], row = i // 3 + 1, col = i % 3 + 1)
            fig.update_yaxes(range = [0, 20], row = i // 3 + 1, col = i % 3 + 1)
            fig.add_vline(x = distribution["Combined Slope"].median(), row = i // 3 + 1, col = i % 3 + 1, line = dict(color = "orange"))
            fig.add_annotation(x = distribution["Combined Slope"].median() + 0.75, y = 18, text = f"{distribution['Combined Slope'].median():.2f}", 
                               showarrow = False, font = dict(color = "orange"), row = i // 3 + 1, col = i % 3 + 1)
        fig.update_layout(title = f"Slope Distribution by Month", height = 750, width = 850, showlegend = False)

        return fig

    def plot_cluster_members_average_slope(self, n_clusters = 6):
        results_df = pd.read_csv(f"clustering_results_{self.dataset}_{self.var}_{n_clusters}.csv")
        results_df["Cluster"] = results_df["Cluster"] + 1

        merged_df = pd.merge(results_df, self.load_data(), on = "Region")
        fig = go.Figure()
        for cluster in sorted(merged_df["Cluster"].unique()):
            cluster_distribution = merged_df[merged_df["Cluster"] == cluster]
            print(cluster_distribution)
            cluster_avg = cluster_distribution.groupby(['Month'])['Slope in Original Units'].mean().reset_index()
            fig.add_trace(go.Histogram(x = cluster_distribution["Slope in Original Units"], name = f"Cluster {cluster}"))
        cluster_avg = merged_df.groupby(['Cluster'])['Slope in Original Units'].mean().reset_index()
        print(cluster_avg)

        # sort by month
        cluster_avg = cluster_avg.sort_values(by = ["Month"], key = lambda x: x.map(lambda y: self.months.index(y)))

        fig = px.line(cluster_avg, x = 'Month', y = 'Slope in Original Units', color = 'Cluster', title = 'Average Slope by Cluster and Month')
        fig.update_layout(title = "Average Slope by Cluster and Month")

        return fig
    
    def dtw_with_clustering(self):
        from dtw import dtw

        clustering_data, clustering_data_state_names = self.preprocess_data_for_clustering()
        
        for i, state in enumerate(clustering_data_state_names):
            for j, state2 in enumerate(clustering_data_state_names):
                if i != j:
                    d = dtw(clustering_data[i, :], clustering_data[j, :])
                    print(d.index1, d.index2)

if __name__ == "__main__":
    # fig = CompareRegressionResults("Regression Results/MERRA2/regression_results-MERRA2-T2MMAX.csv", "Regression Results/ERA5/regression_results-ERA5-rescaled-T2MMAX-merra2_timeframe.csv").combined_validation()
    # fig.show()
    # df = pd.read_csv("full_processed_data.csv")
    # PooledEstimator(df).save_pooled_bootstrap("Regression Results/pooled_bootstrap_results.csv")
    # fig = PlotlySlopeMap("Regression Results/pooled_bootstrap_results.csv").make_figure()
    # fig.show()

    fig = AppFunctionsforPooledData("CT").make_by_temp_plot("MA")
    fig.show()
