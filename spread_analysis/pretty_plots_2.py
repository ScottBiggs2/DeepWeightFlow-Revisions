import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import sys
import glob

def plot_similarity_vs_accuracy_from_csv(csv_file, model_type=None):
    """
    Recreate the scatter plot from similarity_vs_accuracy_data_*.csv
    Excludes Original models to reduce visual clutter.
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return None
        
    df = pd.read_csv(csv_file)
    
    # Extract model_type from filename if not provided
    if model_type is None:
        model_type = csv_file.replace('similarity_vs_accuracy_data_', '').replace('.csv', '')
    
    # Filter out Original models to reduce visual clutter
    filtered_df = df[df['model_type'] != 'Original'].copy()
    
    print(f"Loaded {len(filtered_df)} data points from {csv_file} (excluding Original models)")
    print(f"Model types in data: {filtered_df['model_type'].unique()}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 22})
    # Avoid forcing LaTeX rendering (may not be available in environment)
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Define colors for consistency with original
    colors = {
        'Generated': 'red', 
        'Original + N(0, 0.001)': 'orange',
        'Original + N(0, 0.01)': 'green'
    }
    
    # Define labels with LaTeX formatting
    labels = {
        'Generated': 'Generated',
        'Original + N(0, 0.001)': r'Original + $\mathcal{N}(0, 0.001)$',
        'Original + N(0, 0.01)': r'Original + $\mathcal{N}(0, 0.01)$'
    }
    
    # Plot each model type (excluding Original)
    for model_type_name in filtered_df['model_type'].unique():
        subset = filtered_df[filtered_df['model_type'] == model_type_name]
        color = colors.get(model_type_name, 'purple')
        label = labels.get(model_type_name, model_type_name)
        plt.scatter(subset['max_iou_similarity'], subset['accuracy'], 
                   alpha=0.6, label=label, color=color, s=50)
    
    plt.xlabel('Maximum IoU Similarity vs Original Models')
    plt.ylabel('Test Accuracy (\\%)')
    plt.xlim(0.0, 1)
    plt.ylim(25, 95)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = f'similarity_vs_accuracy_{model_type}_from_csv.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved plot to {output_file}")
    plt.show()
    
    return filtered_df

def plot_violin_plots_from_csv(csv_file, model_type=None):
    """
    Recreate the violin plots from violin_plot_data_*.csv
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return None
        
    df = pd.read_csv(csv_file)
    
    # Extract model_type from filename if not provided
    if model_type is None:
        model_type = csv_file.replace('violin_plot_data_', '').replace('.csv', '')
    
    print(f"Loaded {len(df)} data points from {csv_file}")
    print(f"Categories in data: {df['Category'].unique()}")
    
    # Create the plots
    plt.rcParams.update({'font.size': 22})
    # Avoid forcing LaTeX rendering (may not be available in environment)
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Top plot: Violin plot of accuracies
    sns.violinplot(data=df, x='Category', y='Accuracy', ax=ax1)
    ax1.set_title(f'Test Accuracy Distributions - {model_type}')
    ax1.set_ylabel('Test Accuracy (\\%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Bottom plot: Violin plot of max similarities
    sns.violinplot(data=df, x='Category', y='Max_Similarity', ax=ax2)
    ax2.set_ylabel('Maximum IoU Similarity')
    ax2.set_xlabel('Model Type')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = f'violin_plots_{model_type}_from_csv.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved plot to {output_file}")
    plt.show()
    
    # Print aggregate statistics
    print(f"\n=== Aggregate IoU Statistics vs Original Models for {model_type} ===")
    for category in df['Category'].unique():
        subset = df[df['Category'] == category]['Max_Similarity']
        print(f"{category} - Mean Max IoU: {subset.mean():.4f} ± {subset.std():.4f}")
    
    return df

def plot_noise_comparison_from_csv(csv_file, model_type=None):
    """
    Create a plot comparing original models with different noise levels and generated models.
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return None
        
    df = pd.read_csv(csv_file)
    
    # Extract model_type from filename if not provided
    if model_type is None:
        model_type = csv_file.replace('similarity_vs_accuracy_data_', '').replace('.csv', '')
    
    # Include all model types (Original, noise variants, and Generated)
    print(f"Loaded {len(df)} data points from {csv_file}")
    print(f"Model types in noise comparison: {df['model_type'].unique()}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 22})
    # Avoid forcing LaTeX rendering (may not be available in environment)
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Define colors for consistency with original
    colors = {
        'Original': 'blue',
        'Generated': 'red',
        'Original + N(0, 0.001)': 'orange',
        'Original + N(0, 0.01)': 'green'
    }
    
    # Define labels with LaTeX formatting
    labels = {
        'Original': 'Original',
        'Generated': 'Generated',
        'Original + N(0, 0.001)': r'Original + $\mathcal{N}(0, 0.001)$',
        'Original + N(0, 0.01)': r'Original + $\mathcal{N}(0, 0.01)$'
    }
    
    # Plot each model type
    for model_type_name in df['model_type'].unique():
        subset = df[df['model_type'] == model_type_name]
        color = colors.get(model_type_name, 'purple')
        label = labels.get(model_type_name, model_type_name)
        plt.scatter(subset['max_iou_similarity'], subset['accuracy'], 
                   alpha=0.6, label=label, color=color, s=50)
    
    plt.xlabel('Maximum IoU Similarity vs Original Models')
    plt.ylabel('Test Accuracy (\\%)')
    plt.xlim(0.0, 1.0)
    plt.ylim(25, 95)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f'noise_comparison_{model_type}_from_csv.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved noise comparison plot to {output_file}")
    plt.show()
    
    return df

def plot_generated_vs_noisy_reference_from_csv(csv_file, model_type=None):
    """
    Create a plot comparing Generated models vs Original + N(0, 0.01) reference,
    showing only Generated models (removing reference points to reduce visual clutter).
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return None
        
    df = pd.read_csv(csv_file)
    
    # Extract model_type from filename if not provided
    if model_type is None:
        model_type = csv_file.replace('generated_vs_noisy_reference_data_', '').replace('.csv', '')
    
    # Filter to only show Generated models (remove reference points for clarity)
    generated_df = df[df['model_type'] == 'Generated'].copy()
    
    print(f"Loaded {len(generated_df)} Generated model data points from {csv_file}")
    print("Showing only Generated models (reference points removed for visual clarity)")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 22})
    # Avoid forcing LaTeX rendering (may not be available in environment)
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Plot only Generated models
    plt.scatter(generated_df['max_iou_similarity_vs_noisy_ref'], generated_df['accuracy'], 
               alpha=0.6, label='Generated', color='red', s=50)
    
    plt.xlabel(r'Maximum IoU Similarity vs Original + $\mathcal{N}(0, 0.01)$ Models')
    plt.ylabel('Test Accuracy (\\%)')
    plt.xlim(0.0, 1.0)
    plt.ylim(25, 95)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f'generated_vs_noisy_reference_{model_type}_from_csv.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved Generated vs Noisy Reference plot to {output_file}")
    plt.show()
    
    # Print statistics for Generated models
    print(f"\n=== Generated vs Noisy Reference Statistics for {model_type} ===")
    print(f"Generated - Mean Max IoU vs Noisy Ref: {generated_df['max_iou_similarity_vs_noisy_ref'].mean():.4f} ± {generated_df['max_iou_similarity_vs_noisy_ref'].std():.4f}")
    print(f"Generated - Mean Accuracy: {generated_df['accuracy'].mean():.2f}% ± {generated_df['accuracy'].std():.2f}%")
    
    return generated_df

def plot_source_distribution_from_csv(csv_file, model_type=None):
    """
    Recreate the source distribution comparison plot from source_distribution_comparison_*.csv
    """
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found")
        return None
        
    df = pd.read_csv(csv_file)
    
    # Extract model_type from filename if not provided
    if model_type is None:
        model_type = csv_file.replace('source_distribution_comparison_', '').replace('.csv', '')
    
    print(f"Loaded {len(df)} data points from {csv_file}")
    print(f"Source distributions in data: {df['source_std'].unique()}")
    print(f"Model types in data: {df['model_type'].unique()}")
    
    # Create scatter plot
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 22})
    # Avoid forcing LaTeX rendering (may not be available in environment)
    # plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    # Define colors
    colors = {
        'Original': 'blue',
        'Generated (σ=0.001)': 'red',
        'Generated (σ=0.005)': 'orange', 
        'Generated (σ=0.01)': 'green'
    }
    
    # Define labels with LaTeX formatting
    labels = {
        'Generated (σ=0.001)': r'Generated ($\sigma=0.001$)',
        'Generated (σ=0.005)': r'Generated ($\sigma=0.005$)',
        'Generated (σ=0.01)': r'Generated ($\sigma=0.01$)'
    }
    
    # Plot original models first
    orig_subset = df[df['source_std'] == 'original']
    if len(orig_subset) > 0:
        plt.scatter(orig_subset['max_iou_similarity'], orig_subset['accuracy'], 
                   alpha=0.6, label='Original', color='blue', s=50)
    
    # Plot generated models from different source distributions
    source_stds = [std for std in df['source_std'].unique() if std != 'original']
    for i, source_std in enumerate(source_stds):
        subset = df[df['source_std'] == source_std]
        if len(subset) > 0:
            model_type_name = subset['model_type'].iloc[0]
            color = colors.get(model_type_name, ['red', 'orange', 'green'][i % 3])
            label = labels.get(model_type_name, model_type_name)
            plt.scatter(subset['max_iou_similarity'], subset['accuracy'],
                       alpha=0.6, label=label,
                       color=color, s=50)
            
            # Print statistics
            print(f"Source std {source_std} - Mean accuracy: {subset['accuracy'].mean():.2f}% ± {subset['accuracy'].std():.2f}%")
            print(f"Source std {source_std} - Mean IoU vs orig: {subset['max_iou_similarity'].mean():.4f} ± {subset['max_iou_similarity'].std():.4f}")
    
    plt.xlabel('Maximum IoU Similarity vs Original Models')
    plt.ylabel('Test Accuracy (\\%)')
    plt.xlim(0.0, 1.0)
    plt.ylim(25, 95)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f'source_distribution_comparison_{model_type}_from_csv.pdf'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved plot to {output_file}")
    plt.show()
    
    return df

def list_available_csvs():
    """
    List all CSV files that match the expected patterns
    """
    patterns = [
        'similarity_vs_accuracy_data_*.csv',
        'violin_plot_data_*.csv', 
        'source_distribution_comparison_*.csv',
        'generated_vs_noisy_reference_data_*.csv'
    ]
    
    print("Available CSV files:")
    for pattern in patterns:
        files = glob.glob(pattern)
        if files:
            print(f"\n{pattern}:")
            for f in sorted(files):
                print(f"  - {f}")
        else:
            print(f"\n{pattern}: No files found")

def main():
    parser = argparse.ArgumentParser(description='Plot results from CSV files generated by the main script')
    parser.add_argument('--plot-type', choices=['scatter', 'violin', 'source', 'noise', 'gen-vs-noisy', 'all'], 
                       default='all', help='Type of plot to generate')
    parser.add_argument('--model-type', type=str, 
                       help='Model type (e.g., with_gitrebasin, without_rebasin)')
    parser.add_argument('--list', action='store_true', 
                       help='List available CSV files')
    parser.add_argument('--csv-file', type=str,
                       help='Specific CSV file to plot')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_csvs()
        return
    
    if args.csv_file:
        # Plot from specific CSV file. Respect --plot-type (including 'all') and
        # attempt to find related CSVs for the same model_type when appropriate.
        csv = args.csv_file

        # Helper to extract model_type from a known prefixed filename
        def extract_model_type(name):
            for p in ['similarity_vs_accuracy_data_', 'violin_plot_data_', 'source_distribution_comparison_', 'generated_vs_noisy_reference_data_']:
                if p in name:
                    return name.replace(p, '').replace('.csv', '')
            return None

        model_type_from_csv = extract_model_type(csv) or args.model_type

        called = False

        # If user asked for scatter or all, try to plot similarity scatter
        if args.plot_type in ['scatter', 'all']:
            if 'similarity_vs_accuracy_data' in csv:
                plot_similarity_vs_accuracy_from_csv(csv, model_type_from_csv)
                called = True
            else:
                candidate = f'similarity_vs_accuracy_data_{model_type_from_csv}.csv' if model_type_from_csv else None
                if candidate and os.path.exists(candidate):
                    plot_similarity_vs_accuracy_from_csv(candidate, model_type_from_csv)
                    called = True

        # Noise comparison uses the similarity csv as input
        if args.plot_type in ['noise', 'all']:
            if 'similarity_vs_accuracy_data' in csv:
                plot_noise_comparison_from_csv(csv, model_type_from_csv)
                called = True
            else:
                candidate = f'similarity_vs_accuracy_data_{model_type_from_csv}.csv' if model_type_from_csv else None
                if candidate and os.path.exists(candidate):
                    plot_noise_comparison_from_csv(candidate, model_type_from_csv)
                    called = True

        # Generated vs noisy reference: prefer an explicit file, else try derived name
        if args.plot_type in ['gen-vs-noisy', 'all']:
            if 'generated_vs_noisy_reference_data' in csv:
                plot_generated_vs_noisy_reference_from_csv(csv, model_type_from_csv)
                called = True
            else:
                candidate = f'generated_vs_noisy_reference_data_{model_type_from_csv}.csv' if model_type_from_csv else None
                if candidate and os.path.exists(candidate):
                    plot_generated_vs_noisy_reference_from_csv(candidate, model_type_from_csv)
                    called = True

        # Violin plots come from the violin CSV
        if args.plot_type in ['violin', 'all']:
            if 'violin_plot_data' in csv:
                plot_violin_plots_from_csv(csv, model_type_from_csv)
                called = True
            else:
                candidate = f'violin_plot_data_{model_type_from_csv}.csv' if model_type_from_csv else None
                if candidate and os.path.exists(candidate):
                    plot_violin_plots_from_csv(candidate, model_type_from_csv)
                    called = True

        # Source distribution comparison
        if args.plot_type in ['source', 'all']:
            if 'source_distribution_comparison' in csv:
                plot_source_distribution_from_csv(csv, model_type_from_csv)
                called = True
            else:
                candidate = f'source_distribution_comparison_{model_type_from_csv}.csv' if model_type_from_csv else None
                if candidate and os.path.exists(candidate):
                    plot_source_distribution_from_csv(candidate, model_type_from_csv)
                    called = True

        if not called:
            print(f"No matching plots found for '{csv}' with plot-type '{args.plot_type}'.")
        return
    
    # Auto-detect CSV files and plot
    if args.model_type:
        model_types = [args.model_type]
    else:
        # Try to detect model types from available CSV files
        model_types = []
        for f in os.listdir('.'):
            if 'similarity_vs_accuracy_data_' in f:
                mt = f.replace('similarity_vs_accuracy_data_', '').replace('.csv', '')
                if mt not in model_types:
                    model_types.append(mt)
    
    if not model_types:
        print("No CSV files found. Use --list to see available files.")
        return
    
    for model_type in model_types:
        print(f"\n=== Processing plots for model type: {model_type} ===")
        
        if args.plot_type in ['scatter', 'all']:
            csv_file = f'similarity_vs_accuracy_data_{model_type}.csv'
            if os.path.exists(csv_file):
                print(f"\nCreating scatter plot from {csv_file}")
                plot_similarity_vs_accuracy_from_csv(csv_file, model_type)
            else:
                print(f"Scatter plot CSV not found: {csv_file}")
        
        if args.plot_type in ['noise', 'all']:
            csv_file = f'similarity_vs_accuracy_data_{model_type}.csv'
            if os.path.exists(csv_file):
                print(f"\nCreating noise comparison plot from {csv_file}")
                plot_noise_comparison_from_csv(csv_file, model_type)
            else:
                print(f"Noise comparison CSV not found: {csv_file}")
        
        if args.plot_type in ['gen-vs-noisy', 'all']:
            csv_file = f'generated_vs_noisy_reference_data_{model_type}.csv'
            if os.path.exists(csv_file):
                print(f"\nCreating Generated vs Noisy Reference plot from {csv_file}")
                plot_generated_vs_noisy_reference_from_csv(csv_file, model_type)
            else:
                print(f"Generated vs Noisy Reference CSV not found: {csv_file}")
        
        if args.plot_type in ['violin', 'all']:
            csv_file = f'violin_plot_data_{model_type}.csv'
            if os.path.exists(csv_file):
                print(f"\nCreating violin plots from {csv_file}")
                plot_violin_plots_from_csv(csv_file, model_type)
            else:
                print(f"Violin plot CSV not found: {csv_file}")
        
        if args.plot_type in ['source', 'all']:
            csv_file = f'source_distribution_comparison_{model_type}.csv'
            if os.path.exists(csv_file):
                print(f"\nCreating source distribution plot from {csv_file}")
                plot_source_distribution_from_csv(csv_file, model_type)
            else:
                print(f"Source distribution CSV not found: {csv_file}")

if __name__ == "__main__":
    # If no command line arguments, show usage
    if len(sys.argv) == 1:
        print("Usage examples:")
        print("  python plot_from_csv.py --list                    # List available CSV files")
        print("  python plot_from_csv.py --plot-type all          # Plot all types for all model types")
        print("  python plot_from_csv.py --plot-type scatter      # Only scatter plots")
        print("  python plot_from_csv.py --plot-type noise        # Only noise comparison plots")
        print("  python plot_from_csv.py --plot-type gen-vs-noisy # Generated vs Noisy Reference comparison")
        print("  python plot_from_csv.py --model-type with_gitrebasin  # Specific model type")
        print("  python plot_from_csv.py --csv-file data.csv      # Specific CSV file")
        print()
        list_available_csvs()
    else:
        main()