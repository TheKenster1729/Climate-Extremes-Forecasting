#!/usr/bin/env python3
"""
Script to pre-generate all static HTML plots for the webapp.
Run this once to create all the static files, then the webapp can serve them quickly.
"""

from analysis import AppFunctionsforPooledData

def main():
    print("🚀 Starting static plot pre-generation...")
    
    # Create an instance (scenario doesn't matter for pre-generation)
    app = AppFunctionsforPooledData(scenario="aa")
    
    # Pre-generate all plots
    metadata = app.pregenerate_all_plots(output_dir="webapp_plots")
    
    print(f"\n✅ Successfully generated {metadata['total_plots']} plots!")
    print(f"📊 {len(metadata['states'])} states × {len(metadata['scenarios'])} scenarios × {len(metadata['plot_types'])} plot types")
    print(f"💾 Files saved in 'webapp_plots/' directory")
    print(f"⚡ Your webapp will now load plots in ~1 second!")

if __name__ == "__main__":
    main()