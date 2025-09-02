import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Rectangle

# Data from the intensity table for 0←0 transition
# These values represent the fundamental vibrational transition intensities
intensity_data = {
    'HCl': 0.8934,
    'HF': 0.9124,
    'HBr': 0.8756,
    'HI': 0.8421,
    'CO': 0.9234,
    'NO': 0.8678,
    'O₂': 0.8945,
    'N₂': 0.9178,
    'Cl₂': 0.7892,
    'Br₂': 0.7234
}

# Color scheme for molecules
colors = {
    'HCl': '#FF6B6B',
    'HF': '#4ECDC4',
    'HBr': '#45B7D1',
    'HI': '#96CEB4',
    'CO': '#FFEAA7',
    'NO': '#DDA0DD',
    'O₂': '#98D8C8',
    'N₂': '#F7DC6F',
    'Cl₂': '#BB8FCE',
    'Br₂': '#F8C471'
}

def create_franck_condon_plot():
    """Create comprehensive Franck-Condon intensity plot with analysis"""
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Prepare data
    molecules = list(intensity_data.keys())
    intensities = list(intensity_data.values())
    x_positions = np.arange(len(molecules))
    
    # Plot 1: Line plot with markers
    ax1.plot(x_positions, intensities, 'o-', linewidth=2.5, markersize=8, 
             color='#2563eb', markerfacecolor='white', markeredgewidth=2)
    
    # Add colored markers for each molecule
    for i, (molecule, intensity) in enumerate(intensity_data.items()):
        ax1.scatter(i, intensity, c=colors[molecule], s=100, zorder=5, 
                   edgecolors='white', linewidths=2)
        
        # Add value labels
        ax1.annotate(f'{intensity:.4f}', (i, intensity), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Molecules', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax1.set_title('0←0 Vibrational Transition Intensities (Fundamental)', fontsize=14, pad=20)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(molecules, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0.7, 0.95)
    
    # Add background shading for different intensity ranges
    ax1.axhspan(0.9, 0.95, alpha=0.1, color='green', label='Very High Intensity')
    ax1.axhspan(0.85, 0.9, alpha=0.1, color='yellow', label='High Intensity')
    ax1.axhspan(0.8, 0.85, alpha=0.1, color='orange', label='Medium Intensity')
    ax1.axhspan(0.7, 0.8, alpha=0.1, color='red', label='Lower Intensity')
    
    # Plot 2: Bar chart for comparison
    bars = ax2.bar(x_positions, intensities, color=[colors[mol] for mol in molecules],
                   alpha=0.8, edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Molecules', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Intensity', fontsize=12, fontweight='bold')
    ax2.set_title('Intensity Comparison (Bar Chart)', fontsize=14, pad=20)
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels(molecules, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_ylim(0.7, 0.95)
    
    # Add value labels on bars
    for bar, intensity in zip(bars, intensities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{intensity:.4f}', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def print_analysis():
    """Print detailed analysis of the data"""
    print("="*60)
    print("FRANCK-CONDON 0←0 TRANSITION INTENSITY ANALYSIS")
    print("="*60)
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(list(intensity_data.items()), 
                      columns=['Molecule', 'Intensity'])
    df = df.sort_values('Intensity', ascending=False).reset_index(drop=True)
    
    print("\nRanked Intensity Data:")
    print("-" * 40)
    max_intensity = df['Intensity'].max()
    
    for i, row in df.iterrows():
        molecule, intensity = row['Molecule'], row['Intensity']
        relative_percent = (intensity / max_intensity) * 100
        print(f"{i+1:2d}. {molecule:4s}: {intensity:.4f} ({relative_percent:.1f}% of max)")
    
    print(f"\nStatistical Summary:")
    print("-" * 30)
    print(f"Highest Intensity: {df.iloc[0]['Molecule']} ({df['Intensity'].max():.4f})")
    print(f"Lowest Intensity:  {df.iloc[-1]['Molecule']} ({df['Intensity'].min():.4f})")
    print(f"Range:             {df['Intensity'].max() - df['Intensity'].min():.4f}")
    print(f"Average:           {df['Intensity'].mean():.4f}")
    print(f"Standard Deviation: {df['Intensity'].std():.4f}")
    
    # Group analysis
    print(f"\nMolecular Type Analysis:")
    print("-" * 30)
    
    # Hydrogen halides
    h_halides = ['HF', 'HCl', 'HBr', 'HI']
    h_halide_intensities = [intensity_data[mol] for mol in h_halides if mol in intensity_data]
    print(f"Hydrogen Halides (HX): Avg = {np.mean(h_halide_intensities):.4f}")
    
    # Diatomic homonuclear
    homonuclear = ['O₂', 'N₂', 'Cl₂', 'Br₂']
    homonuclear_intensities = [intensity_data[mol] for mol in homonuclear if mol in intensity_data]
    print(f"Homonuclear Diatomics: Avg = {np.mean(homonuclear_intensities):.4f}")
    
    # Heteronuclear (CO, NO)
    heteronuclear = ['CO', 'NO']
    heteronuclear_intensities = [intensity_data[mol] for mol in heteronuclear if mol in intensity_data]
    print(f"Other Heteronuclear:   Avg = {np.mean(heteronuclear_intensities):.4f}")
    
    print(f"\nKey Observations for 0←0 Transition:")
    print("-" * 40)
    print("• 0←0 transitions represent the fundamental vibrational mode")
    print("• Generally show higher intensities compared to higher transitions")
    print("• Reflect the overlap of ground state vibrational wavefunctions")
    print("• Most allowed transition in vibrational spectroscopy")

def save_data_to_csv():
    """Save the data to a CSV file"""
    df = pd.DataFrame(list(intensity_data.items()), 
                      columns=['Molecule', 'Intensity_0_to_0'])
    
    # Add additional calculated columns
    max_intensity = df['Intensity_0_to_0'].max()
    df['Relative_Percentage'] = (df['Intensity_0_to_0'] / max_intensity * 100).round(1)
    df['Molecular_Type'] = df['Molecule'].apply(classify_molecule)
    
    # Sort by intensity
    df = df.sort_values('Intensity_0_to_0', ascending=False)
    
    # Save to CSV
    df.to_csv('franck_condon_0_to_0_intensities.csv', index=False)
    print(f"\nData saved to 'franck_condon_0_to_0_intensities.csv'")
    return df

def classify_molecule(molecule):
    """Classify molecules by type"""
    if molecule in ['HF', 'HCl', 'HBr', 'HI']:
        return 'Hydrogen Halide'
    elif molecule in ['O₂', 'N₂', 'Cl₂', 'Br₂']:
        return 'Homonuclear Diatomic'
    elif molecule in ['CO', 'NO']:
        return 'Heteronuclear Diatomic'
    else:
        return 'Other'

def create_comparison_plot():
    """Create additional comparison plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detailed Franck-Condon 0←0 Analysis', fontsize=16, fontweight='bold')
    
    # Data preparation
    df = pd.DataFrame(list(intensity_data.items()), 
                      columns=['Molecule', 'Intensity'])
    df['Molecular_Type'] = df['Molecule'].apply(classify_molecule)
    df_sorted = df.sort_values('Intensity', ascending=False)
    
    # Plot 1: Horizontal bar chart
    ax1 = axes[0, 0]
    y_pos = np.arange(len(df_sorted))
    bars = ax1.barh(y_pos, df_sorted['Intensity'], 
                    color=[colors[mol] for mol in df_sorted['Molecule']], alpha=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(df_sorted['Molecule'])
    ax1.set_xlabel('Intensity')
    ax1.set_title('Ranked by Intensity')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Molecular type comparison
    ax2 = axes[0, 1]
    type_groups = df.groupby('Molecular_Type')['Intensity'].agg(['mean', 'std'])
    type_groups.plot(kind='bar', y='mean', yerr='std', ax=ax2, 
                     color=['#FF9999', '#66B2FF', '#99FF99'], alpha=0.8)
    ax2.set_title('Average Intensity by Molecular Type')
    ax2.set_ylabel('Average Intensity')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Distribution histogram
    ax3 = axes[1, 0]
    ax3.hist(df['Intensity'], bins=8, alpha=0.7, color='lightgreen', 
             edgecolor='black', linewidth=1)
    ax3.set_xlabel('Intensity')
    ax3.set_ylabel('Frequency')
    ax3.set_title('0←0 Intensity Distribution')
    ax3.axvline(df['Intensity'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["Intensity"].mean():.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot with molecular type coloring
    ax4 = axes[1, 1]
    type_colors = {'Hydrogen Halide': 'red', 'Homonuclear Diatomic': 'blue', 
                   'Heteronuclear Diatomic': 'green'}
    for mol_type in df['Molecular_Type'].unique():
        subset = df[df['Molecular_Type'] == mol_type]
        ax4.scatter(range(len(subset)), subset['Intensity'], 
                   c=type_colors[mol_type], label=mol_type, s=100, alpha=0.7)
    
    ax4.set_xlabel('Molecule Index')
    ax4.set_ylabel('Intensity')
    ax4.set_title('0←0 Intensity by Molecular Type')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Additional utility functions
def compare_molecules(mol1, mol2):
    """Compare two specific molecules"""
    if mol1 not in intensity_data or mol2 not in intensity_data:
        print("One or both molecules not found in data")
        return
    
    int1, int2 = intensity_data[mol1], intensity_data[mol2]
    diff = abs(int1 - int2)
    percent_diff = (diff / max(int1, int2)) * 100
    
    print(f"\nComparison: {mol1} vs {mol2}")
    print(f"{mol1}: {int1:.4f}")
    print(f"{mol2}: {int2:.4f}")
    print(f"Difference: {diff:.4f} ({percent_diff:.1f}%)")
    
    if int1 > int2:
        print(f"{mol1} has {diff:.4f} higher intensity than {mol2}")
    else:
        print(f"{mol2} has {diff:.4f} higher intensity than {mol1}")

def find_similar_intensities(target_intensity, tolerance=0.02):
    """Find molecules with similar intensities"""
    similar = []
    for molecule, intensity in intensity_data.items():
        if abs(intensity - target_intensity) <= tolerance:
            similar.append((molecule, intensity))
    
    return sorted(similar, key=lambda x: abs(x[1] - target_intensity))

def analyze_transition_characteristics():
    """Analyze specific characteristics of 0←0 transitions"""
    print(f"\n0←0 TRANSITION CHARACTERISTICS:")
    print("-" * 40)
    print("• Fundamental vibrational transition (v'=0 → v''=0)")
    print("• Typically has the highest Franck-Condon factor")
    print("• Represents the most probable vibrational transition")
    print("• Intensity depends on vibrational wavefunction overlap")
    print("• All values should be relatively high (>0.7 typically)")
    
    # Check for any unusually low values
    df = pd.DataFrame(list(intensity_data.items()), 
                      columns=['Molecule', 'Intensity'])
    low_intensity = df[df['Intensity'] < 0.8]
    
    if not low_intensity.empty:
        print(f"\nMolecules with lower than expected 0←0 intensity (<0.8):")
        for _, row in low_intensity.iterrows():
            print(f"  {row['Molecule']}: {row['Intensity']:.4f}")
    else:
        print(f"\nAll molecules show expected high 0←0 transition intensities (≥0.8)")

def run_additional_analysis():
    """Run additional analysis functions"""
    print("\n" + "="*60)
    print("ADDITIONAL ANALYSIS")
    print("="*60)
    
    # Compare specific molecules
    compare_molecules('N₂', 'CO')
    compare_molecules('HF', 'HCl')
    compare_molecules('Cl₂', 'Br₂')
    
    # Find molecules with similar intensities to CO
    print(f"\nMolecules similar to CO (0.9234 ± 0.02):")
    similar_to_co = find_similar_intensities(0.9234, 0.02)
    for mol, intensity in similar_to_co:
        print(f"  {mol}: {intensity:.4f}")
    
    # Analyze transition characteristics
    analyze_transition_characteristics()

# Main execution
if __name__ == "__main__":
    # Print analysis
    print_analysis()
    
    # Create and show main plot
    fig1 = create_franck_condon_plot()
    plt.show()
    
    # Create comparison plots
    fig2 = create_comparison_plot()
    plt.show()
    
    # Save data
    df = save_data_to_csv()
    print("\nDataFrame Preview:")
    print(df)
    
    # Run additional analysis
    run_additional_analysis()
    
    # Optional: Save plots
    fig1.savefig('franck_condon_0_to_0_main_plot.png', dpi=300, bbox_inches='tight')
    fig2.savefig('franck_condon_0_to_0_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlots saved as 'franck_condon_0_to_0_main_plot.png' and 'franck_condon_0_to_0_analysis.png'")
    
    # Interactive plot option (requires plotly)
    try:
        import plotly.graph_objects as go
        
        # Create interactive plot
        fig_interactive = go.Figure()
        
        molecules = list(intensity_data.keys())
        intensities = list(intensity_data.values())
        
        fig_interactive.add_trace(go.Scatter(
            x=molecules,
            y=intensities,
            mode='lines+markers',
            line=dict(color='#2563eb', width=3),
            marker=dict(
                size=12,
                color=[colors[mol] for mol in molecules],
                line=dict(color='white', width=2)
            ),
            text=[f'{mol}: {intensity:.4f}' for mol, intensity in intensity_data.items()],
            hovertemplate='<b>%{text}</b><br>Intensity: %{y:.4f}<extra></extra>'
        ))
        
        fig_interactive.update_layout(
            title='Interactive Franck-Condon 0←0 Transition Intensities',
            xaxis_title='Molecules',
            yaxis_title='Intensity',
            hovermode='x',
            template='plotly_white',
            height=500
        )
        
        fig_interactive.show()
        print("\nInteractive plot created (if plotly is installed)")
        
    except ImportError:
        print("\nNote: Install plotly for interactive plots: pip install plotly")

def print_analysis():
    """Print detailed analysis of the data"""
    print("="*60)
    print("FRANCK-CONDON 0←0 TRANSITION INTENSITY ANALYSIS")
    print("="*60)
    
    # Create DataFrame for easier analysis
    df = pd.DataFrame(list(intensity_data.items()), 
                      columns=['Molecule', 'Intensity'])
    df = df.sort_values('Intensity', ascending=False).reset_index(drop=True)
    
    print("\nRanked Intensity Data:")
    print("-" * 40)
    max_intensity = df['Intensity'].max()
    
    for i, row in df.iterrows():
        molecule, intensity = row['Molecule'], row['Intensity']
        relative_percent = (intensity / max_intensity) * 100
        print(f"{i+1:2d}. {molecule:4s}: {intensity:.4f} ({relative_percent:.1f}% of max)")
    
    print(f"\nStatistical Summary:")
    print("-" * 30)
    print(f"Highest Intensity: {df.iloc[0]['Molecule']} ({df['Intensity'].max():.4f})")
    print(f"Lowest Intensity:  {df.iloc[-1]['Molecule']} ({df['Intensity'].min():.4f})")
    print(f"Range:             {df['Intensity'].max() - df['Intensity'].min():.4f}")
    print(f"Average:           {df['Intensity'].mean():.4f}")
    print(f"Standard Deviation: {df['Intensity'].std():.4f}")
    
    # Group analysis
    print(f"\nMolecular Type Analysis:")
    print("-" * 30)
    
    # Hydrogen halides
    h_halides = ['HF', 'HCl', 'HBr', 'HI']
    h_halide_intensities = [intensity_data[mol] for mol in h_halides if mol in intensity_data]
    print(f"Hydrogen Halides (HX): Avg = {np.mean(h_halide_intensities):.4f}")
    
    # Diatomic homonuclear
    homonuclear = ['O₂', 'N₂', 'Cl₂', 'Br₂']
    homonuclear_intensities = [intensity_data[mol] for mol in homonuclear if mol in intensity_data]
    print(f"Homonuclear Diatomics: Avg = {np.mean(homonuclear_intensities):.4f}")
    
    # Heteronuclear (CO, NO)
    heteronuclear = ['CO', 'NO']
    heteronuclear_intensities = [intensity_data[mol] for mol in heteronuclear if mol in intensity_data]
    print(f"Other Heteronuclear:   Avg = {np.mean(heteronuclear_intensities):.4f}")
    
    print(f"\nKey Observations for 0←0 Transition:")
    print("-" * 40)
    print("• 0←0 transitions represent the fundamental vibrational mode")
    print("• Generally show higher intensities compared to higher transitions")
    print("• Reflect the overlap of ground state vibrational wavefunctions")
    print("• Most allowed transition in vibrational spectroscopy")
    print("• Higher intensities indicate better wavefunction overlap")
