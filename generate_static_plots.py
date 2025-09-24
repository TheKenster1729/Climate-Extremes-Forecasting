#!/usr/bin/env python3
"""
Script to pre-generate all static HTML plots for the webapp.
Run this once to create all the static files, then the webapp can serve them quickly.
Supports all temperature variables: T2MMAX (Max), T2MMEAN (Mean), T2MMIN (Min)
"""

from analysis import AppFunctionsforPooledData

def main():
    print("Starting static plot pre-generation for ALL temperature variables...")
    print("Variables: T2MMAX (Max), T2MMEAN (Mean), T2MMIN (Min)")
    print("Directory structure: webapp_plots/{scenario}/{variable}/{state}_{plot_type}.html")
    print("All plots now include both Confidence Intervals (CI) and Prediction Intervals (PI)")
    
    # Create an instance (scenario doesn't matter for pre-generation)
    app = AppFunctionsforPooledData(scenario="aa", var="T2MMAX")
    
    # Pre-generate plots for all variables
    metadata = app.pregenerate_all_plots(output_dir="webapp_plots", variables=["T2MMAX", "T2MMEAN", "T2MMIN"])
    
    print(f"\nSuccessfully generated {metadata['total_plots']} plots!")
    print(f"{len(metadata['states'])} states x {len(metadata['scenarios'])} scenarios x {len(metadata['variables'])} variables x {len(metadata['plot_types'])} plot types")
    print(f"Files saved in 'webapp_plots/' directory with structure:")
    print(f"   webapp_plots/")
    print(f"   ├── aa/")
    print(f"   │   ├── Max/     (T2MMAX plots with CI + PI)")
    print(f"   │   ├── Mean/    (T2MMEAN plots with CI + PI)")
    print(f"   │   └── Min/     (T2MMIN plots with CI + PI)")
    print(f"   └── ct/")
    print(f"       ├── Max/     (T2MMAX plots with CI + PI)")
    print(f"       ├── Mean/    (T2MMEAN plots with CI + PI)")
    print(f"       └── Min/     (T2MMIN plots with CI + PI)")
    print(f"Your webapp will now load all temperature variable plots in ~1 second!")
    print(f"All plots now have consistent styling with both confidence and prediction intervals!")

if __name__ == "__main__":
    main()