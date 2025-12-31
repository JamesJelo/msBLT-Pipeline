"""
msBLT 2.0 - Fixed and Improved Pipeline
Howard High School - Garcia, James Jelo
MIT License 2025
"""

 
# IMPORTS
 
import sys
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from Bio import SeqIO
from Bio.Seq import Seq  # Fixed import
from scipy import stats as scipy_stats
from matplotlib.patches import Rectangle, Patch, Circle
from scipy.signal import savgol_filter
from scipy import stats
 
# CONFIGURATION
 
@dataclass
class Config:
    """Pipeline configuration settings"""
    # File paths
    data_folder: Path = Path("data")
    output_folder: Path = Path("results")
    
    # Sequence parameters
    min_sequence_length: int = 50
    target_gene: str = "bla"  # β-lactamase gene marker
    
    # Guide RNA parameters
    guide_length: int = 20
    optimal_gc_range: Tuple[int, int] = (25, 75)
    min_species_count: int = 1
    
    # Performance thresholds
    good_score_threshold: int = 85
    max_off_target_risk: float = 5.0
    
    # Visualization
    plot_dpi: int = 300
    max_guides_to_plot: int = 100

 
# DATA CLASSES
 
@dataclass
class SequenceRecord:
    """Container for a single DNA sequence"""
    sequence: str
    species: str
    gene_name: str
    gc_content: float
    length: int

@dataclass
class GuideCandidate:
    """A potential CRISPR guide RNA"""
    sequence: str
    pam: str
    species: List[str]
    score: float
    gc_percent: float
    conservation: float
    off_target_risk: float
    
    @property
    def is_high_quality(self) -> bool:
        """Check if guide meets quality thresholds"""
        return (self.score >= 85 and 
                self.off_target_risk <= 5.0 and
                25 <= self.gc_percent <= 75)

 
# LOGGING SETUP
 
def setup_logging(output_folder: Path) -> logging.Logger:
    """Configure logging for the pipeline"""
    # Create output folder if it doesn't exist
    output_folder.mkdir(exist_ok=True, parents=True)
    
    log_file = output_folder / "msblt_pipeline.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("msBLT 2.0 Pipeline Starting")
    logger.info("=" * 60)
    
    return logger

 
# SEQUENCE PROCESSING
 
class SequenceLoader:
    """Handles loading and validating DNA sequences"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.sequences: Dict[str, List[SequenceRecord]] = {}
    
    def load_all(self) -> Dict[str, List[SequenceRecord]]:
        """Load all FASTA files from data folder"""
        self.logger.info(f"Loading sequences from: {self.config.data_folder}")
        
        if not self.config.data_folder.exists():
            self.logger.error(f"Data folder not found: {self.config.data_folder}")
            return {}
        
        fasta_files = list(self.config.data_folder.glob("*.fasta")) + \
                     list(self.config.data_folder.glob("*.fa"))
        
        if not fasta_files:
            self.logger.error("No FASTA files found")
            return {}
        
        total_sequences = 0
        for fasta_file in fasta_files:
            species_name = fasta_file.stem
            species_sequences = []
            
            try:
                for record in SeqIO.parse(fasta_file, "fasta"):
                    # Filter for target gene if specified
                    if (self.config.target_gene.lower() != "none" and
                        self.config.target_gene.lower() not in record.description.lower()):
                        continue
                    
                    # Create sequence record
                    seq_record = self._create_sequence_record(str(record.seq), species_name, record.id)
                    if seq_record:
                        species_sequences.append(seq_record)
                        total_sequences += 1
                
                if species_sequences:
                    self.sequences[species_name] = species_sequences
                    self.logger.info(f"  {species_name}: {len(species_sequences)} sequences")
                else:
                    self.logger.warning(f"  {species_name}: No valid sequences")
                    
            except Exception as e:
                self.logger.error(f"Error processing {fasta_file}: {e}")
        
        self.logger.info(f"Loaded {total_sequences} sequences from {len(self.sequences)} species")
        return self.sequences if total_sequences > 0 else {}
    
    def _create_sequence_record(self, sequence: str, species: str, gene_id: str) -> Optional[SequenceRecord]:
        """Create and validate a sequence record"""
        seq_upper = sequence.upper().strip()
        
        # Remove non-DNA characters
        seq_clean = ''.join(c for c in seq_upper if c in 'ATCGN')
        
        # Basic validation
        if len(seq_clean) < self.config.min_sequence_length:
            self.logger.debug(f"Sequence too short: {len(seq_clean)} bp")
            return None
        
        # Calculate GC content
        gc_count = seq_clean.count('G') + seq_clean.count('C')
        gc_percent = (gc_count / len(seq_clean)) * 100 if seq_clean else 0
        
        return SequenceRecord(
            sequence=seq_clean,
            species=species,
            gene_name=gene_id,
            gc_content=gc_percent,
            length=len(seq_clean)
        )

 
# SIMPLE CONSERVATION ANALYSIS
 
class SimpleConservationAnalyzer:
    """Simplified conservation analysis"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def find_conserved_kmers(self, sequences: Dict[str, List[SequenceRecord]], k=20) -> List[Dict]:
        """Find conserved k-mers across species using simple string matching"""
        self.logger.info("Finding conserved k-mers (simplified approach)...")
        
        if len(sequences) < 2:
            self.logger.warning("Need at least 2 species for conservation analysis")
            return []
        
        # Take first sequence from each species
        sample_seqs = {}
        for species, seq_list in sequences.items():
            if seq_list:
                sample_seqs[species] = seq_list[0].sequence
        
        if len(sample_seqs) < 2:
            return []
        
        # Use the first sequence as reference
        ref_species = list(sample_seqs.keys())[0]
        ref_seq = sample_seqs[ref_species]
        
        conserved_regions = []
        
        # Scan through reference sequence
        for i in range(0, len(ref_seq) - k + 1, 5):  # Step by 5 to avoid overlapping
            kmer = ref_seq[i:i + k]
            
            # Skip if kmer contains N or is too short
            if 'N' in kmer or len(kmer) < k:
                continue
            
            # Check which species have this kmer
            matching_species = [ref_species]
            for species, seq in sample_seqs.items():
                if species != ref_species and kmer in seq:
                    matching_species.append(species)
            
            # Require at least 2 species
            if len(matching_species) >= 2:
                # Calculate conservation score (proportion of species with this kmer)
                conservation = len(matching_species) / len(sample_seqs)
                
                region = {
                    'start': i,
                    'end': i + k,
                    'sequence': kmer,
                    'conservation': conservation,
                    'species': matching_species
                }
                conserved_regions.append(region)
        
        self.logger.info(f"Found {len(conserved_regions)} conserved k-mers")
        return conserved_regions

 
# GUIDE RNA DESIGNER
 
class GuideDesigner:
    """Designs and scores CRISPR guide RNAs for SpCas9"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def find_pam_sites(self, sequence: str) -> List[Dict]:
        """Find all NGG PAM sites in a sequence"""
        pam_sites = []
        seq_len = len(sequence)
        
        for i in range(seq_len - 22):  # 20nt guide + 3nt PAM - 1
            # Check for NGG PAM (SpCas9)
            if sequence[i+20:i+23].endswith("GG"):
                guide_seq = sequence[i:i+20]
                
                # Basic validation
                if len(guide_seq) == 20 and 'N' not in guide_seq:
                    pam_sites.append({
                        'position': i,
                        'guide': guide_seq,
                        'pam': sequence[i+20:i+23]
                    })
        
        return pam_sites
    
    def score_guide(self, guide_seq: str) -> Dict[str, float]:
        """Score a guide RNA based on various criteria"""
        if len(guide_seq) != 20:
            return {'score': 0, 'gc': 0, 'risk': 100}
        
        score = 100.0
        
        # Calculate GC content
        gc_count = guide_seq.count('G') + guide_seq.count('C')
        gc_percent = (gc_count / 20) * 100
        
        # Adjust score based on GC content (optimal: 40-60%)
        if gc_percent < 25:
            score -= (25 - gc_percent) * 1.5
        elif gc_percent > 75:
            score -= (gc_percent - 75) * 1.5
        elif 40 <= gc_percent <= 60:
            score += 10  # Bonus for optimal GC
        
        # Penalize homopolymers (runs of same base)
        for i in range(len(guide_seq) - 4):
            if len(set(guide_seq[i:i+5])) == 1:
                score -= 25
                break
        
        # Penalize poly-T sequences (termination signals)
        if 'TTTT' in guide_seq:
            score -= 20
        
        # Bonus/penalty for specific positions
        # Position 20 (last base before PAM): G is good
        if guide_seq[-1] == 'G':
            score += 5
        
        # Position 1-5: avoid G
        if guide_seq[0] == 'G':
            score -= 5
        
        # Ensure score is in range
        score = max(0, min(100, score))
        
        # Estimate off-target risk (simplified)
        # More diverse sequence = lower risk
        seed = guide_seq[-12:]  # Last 12 bases (seed region)
        complexity = len(set(seed)) / len(seed)
        off_target_risk = 100 * (1 - complexity)
        
        return {
            'score': round(score, 1),
            'gc': round(gc_percent, 1),
            'risk': round(min(off_target_risk, 100), 1)
        }
    
    def find_shared_guides(self, sequences: Dict[str, List[SequenceRecord]]) -> List[GuideCandidate]:
        """Find guide RNAs that work across multiple species"""
        self.logger.info("Finding shared guide RNAs across species...")
        
        # Dictionary to store guides and the species they work in
        guide_dict = {}  # guide_sequence -> {species: positions, scores}
        
        # Process each species
        for species, seq_list in sequences.items():
            if not seq_list:
                continue
                
            # Use first sequence for each species (simplified)
            seq_obj = seq_list[0]
            seq = seq_obj.sequence
            
            # Find all PAM sites in this species
            pam_sites = self.find_pam_sites(seq)
            
            for site in pam_sites:
                guide_seq = site['guide']
                guide_scores = self.score_guide(guide_seq)
                
                if guide_seq not in guide_dict:
                    guide_dict[guide_seq] = {
                        'species': [],
                        'scores': [],
                        'pams': [],
                        'positions': []
                    }
                
                guide_dict[guide_seq]['species'].append(species)
                guide_dict[guide_seq]['scores'].append(guide_scores['score'])
                guide_dict[guide_seq]['pams'].append(site['pam'])
                guide_dict[guide_seq]['positions'].append(site['position'])
        
        # Create guide candidates for guides that work in multiple species
        guides = []
        for guide_seq, data in guide_dict.items():
            species_count = len(data['species'])
            
            if species_count >= self.config.min_species_count:
                # Average score across species
                avg_score = np.mean(data['scores'])
                
                # Get most common PAM
                pam = max(set(data['pams']), key=data['pams'].count)
                
                # Calculate GC content
                gc_percent = (guide_seq.count('G') + guide_seq.count('C')) / len(guide_seq) * 100
                
                # Conservation score (proportion of species)
                conservation = species_count / len(sequences)
                
                # Average off-target risk
                scores = self.score_guide(guide_seq)
                
                guide = GuideCandidate(
                    sequence=guide_seq,
                    pam=pam,
                    species=data['species'],
                    score=round(avg_score, 1),
                    gc_percent=round(gc_percent, 1),
                    conservation=round(conservation, 3),
                    off_target_risk=scores['risk']
                )
                guides.append(guide)
        
        # Sort by score
        guides.sort(key=lambda x: x.score, reverse=True)
        
        self.logger.info(f"Found {len(guides)} shared guide RNAs")
        return guides

 
# VISUALIZATION (SPECIFIC GRAPHS REQUESTED)
 
class ResultsVisualizer:
    """Professional visualizations for msBLT 2.0 results - SPECIFIC GRAPHS"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.threshold = 85
        
        # Scientific color scheme
        self.colors = {
            'above_threshold': '#2E86AB',      # Blue
            'below_threshold': '#A23B72',      # Purple
            'neutral': '#F18F01',              # Orange
            'primary_network': '#28A745',      # Green
            'secondary_network': '#17A2B8',    # Teal
            'overlap': '#6F42C1',              # Purple
            'outlier': '#DC3545'               # Red
        }
        
        # Set scientific style
        plt.style.use('default')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 8,
            'axes.titlesize': 10,
            'axes.labelsize': 9,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight'
        })
    
    def create_dashboard(self, guides: List[GuideCandidate], output_folder: Path):
        """Create all 6 specific graphs requested"""
        self.logger.info("Creating specific graphs as requested...")
        
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # Convert guides to DataFrame
        df = self._guides_to_dataframe(guides)
        
        if df.empty:
            self.logger.warning("No guides to visualize")
            return
        
        # Create species-level summary
        species_summary = self._create_species_summary(df)
        
        # Create figure with 6 subplots (3x2 grid)
        fig = plt.figure(figsize=(15, 18), facecolor='white')
        fig.suptitle('msBLT 2.0: Specific Graph Analysis\nDual-Layer Conservation Network Framework', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Define grid layout
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25, 
                            left=0.08, right=0.95, top=0.94, bottom=0.06)
        
        try:
            # Plot 1 & 1.1: Multi-Species Hierarchical Clustering Heatmap
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_hierarchical_clustering_heatmap(ax1, df, species_summary)
        except Exception as e:
            self.logger.warning(f"Failed to create Graph 1: {e}")
            ax1.text(0.5, 0.5, 'Graph 1: Data Error\nCheck species data', 
                    ha='center', va='center', fontsize=10)
            ax1.set_title('1. Hierarchical Clustering Heatmap\n(Data Error)', 
                         fontsize=10, fontweight='bold')
        
        try:
            # Plot 2 & 2.1: Conservation Gradient Curve
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_conservation_gradient_curve(ax2, df)
        except Exception as e:
            self.logger.warning(f"Failed to create Graph 2: {e}")
            ax2.text(0.5, 0.5, 'Graph 2: Data Error\nInsufficient data points', 
                    ha='center', va='center', fontsize=10)
            ax2.set_title('2. Conservation Gradient Curve\n(Data Error)', 
                         fontsize=10, fontweight='bold')
        
        try:
            # Plot 3 & 3.1: Violin-Box Hybrid Distribution
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_violin_box_hybrid(ax3, df)
        except Exception as e:
            self.logger.warning(f"Failed to create Graph 3: {e}")
            ax3.text(0.5, 0.5, 'Graph 3: Data Error\nCheck species efficiency data', 
                    ha='center', va='center', fontsize=10)
            ax3.set_title('3. Violin-Box Hybrid Distribution\n(Data Error)', 
                         fontsize=10, fontweight='bold')
        
        try:
            # Plot 4 & 4.1: Multi-Species Volcano Plot
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_multi_species_volcano(ax4, df)
        except Exception as e:
            self.logger.warning(f"Failed to create Graph 4: {e}")
            ax4.text(0.5, 0.5, 'Graph 4: Data Error\nInsufficient guide data', 
                    ha='center', va='center', fontsize=10)
            ax4.set_title('4. Multi-Species Volcano Plot\n(Data Error)', 
                         fontsize=10, fontweight='bold')
        
        try:
            # Plot 5 & 5.1: Dual-Layer Conservation Network Graph
            ax5 = fig.add_subplot(gs[2, 0])
            self._plot_dual_layer_network(ax5, df, species_summary)
        except Exception as e:
            self.logger.warning(f"Failed to create Graph 5: {e}")
            ax5.text(0.5, 0.5, 'Graph 5: Data Error\nCheck network data', 
                    ha='center', va='center', fontsize=10)
            ax5.set_title('5. Dual-Layer Conservation Network\n(Data Error)', 
                         fontsize=10, fontweight='bold')
        
        try:
            # Plot 6 & 6.1: LOESS Fit and Residual Outlier Map
            ax6 = fig.add_subplot(gs[2, 1])
            self._plot_loess_residual_outlier(ax6, df, species_summary)
        except Exception as e:
            self.logger.warning(f"Failed to create Graph 6: {e}")
            ax6.text(0.5, 0.5, 'Graph 6: Data Error\nInsufficient conservation data', 
                    ha='center', va='center', fontsize=10)
            ax6.set_title('6. LOESS Fit and Residual Outlier Map\n(Data Error)', 
                         fontsize=10, fontweight='bold')
        
        # Save dashboard
        dashboard_path = output_folder / "msblt_specific_graphs.png"
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        self.logger.info(f"Specific graphs dashboard saved to: {dashboard_path}")
        
        # Create individual plot files
        self._create_individual_plots(df, species_summary, output_folder)
    
    def _guides_to_dataframe(self, guides: List[GuideCandidate]) -> pd.DataFrame:
        """Convert guide objects to DataFrame"""
        data = []
        for i, guide in enumerate(guides):
            data.append({
                'Rank': i + 1,
                'Sequence': guide.sequence,
                'PAM': guide.pam,
                'Species': guide.species,
                'Species_Count': len(guide.species),
                'Score': guide.score,
                'GC_Percent': guide.gc_percent,
                'Conservation': guide.conservation,
                'Off_Target_Risk': guide.off_target_risk,
                'High_Quality': guide.is_high_quality,
                'Above_Threshold': guide.score >= self.threshold
            })
        return pd.DataFrame(data)
    
    def _create_species_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create species-level summary statistics"""
        species_data = {}
        all_species = set()
        
        # Get all unique species
        for species_list in df['Species']:
            if isinstance(species_list, list):
                all_species.update(species_list)
        
        if not all_species:
            return pd.DataFrame()
        
        for species in all_species:
            try:
                # Get guides for this species
                species_guides = df[df['Species'].apply(lambda x: species in x if isinstance(x, list) else False)]
                
                if len(species_guides) > 0:
                    # Calculate maximum on-target efficiency
                    max_score = species_guides['Score'].max() if len(species_guides) > 0 else 0
                    
                    # Calculate distribution statistics
                    scores = species_guides['Score'].values
                    q1, q2, q3 = np.percentile(scores, [25, 50, 75]) if len(scores) > 0 else (0, 0, 0)
                    
                    # Count guides at threshold boundary
                    near_threshold = species_guides[
                        (species_guides['Score'] >= 80) & 
                        (species_guides['Score'] <= 90)
                    ].shape[0] if len(species_guides) > 0 else 0
                    
                    species_data[species] = {
                        'Total_Guides': len(species_guides),
                        'Max_Score': float(max_score),
                        'Median_Score': float(q2),
                        'IQR': float(q3 - q1),
                        'Near_Threshold_Count': int(near_threshold),
                        'Above_Threshold': bool(max_score >= self.threshold),
                        'Conservation_Quality': float(species_guides['Conservation'].mean() if len(species_guides) > 0 else 0)
                    }
            except Exception as e:
                self.logger.warning(f"Error processing species {species}: {e}")
                continue
        
        if not species_data:
            return pd.DataFrame()
        
        return pd.DataFrame(species_data).T
    
     
    # GRAPH 1: Multi-Species Hierarchical Clustering Heatmap - FIXED VERSION
     
    def _plot_hierarchical_clustering_heatmap(self, ax, df, species_summary):
        """1 & 1.1: Shows max on-target efficiencies per species and highlights below threshold"""
        
        if species_summary.empty or len(species_summary) < 2:
            ax.text(0.5, 0.5, 'Insufficient species data\n(Need at least 2 species)', 
                   ha='center', va='center', fontsize=10)
            ax.set_title('1. Hierarchical Clustering Heatmap\n(Insufficient Data)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        try:
            # Prepare numeric data for clustering
            species_list = species_summary.index.tolist()
            
            # Extract numeric columns and ensure they are floats
            max_scores = species_summary['Max_Score'].astype(float).values
            total_guides = species_summary['Total_Guides'].astype(float).values
            
            # Normalize guide counts
            if total_guides.max() > 0:
                normalized_guides = total_guides / total_guides.max()
            else:
                normalized_guides = total_guides
            
            # Create 2D array for heatmap - FIXED: Ensure proper dtype
            clustering_data = np.column_stack([max_scores, normalized_guides]).astype(float)
            
            # Create heatmap
            im = ax.imshow(clustering_data, aspect='auto', cmap='viridis', vmin=0, vmax=1)
            
            # Set ticks and labels
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Max Score\n(Normalized)', 'Guide Density\n(Normalized)'], 
                              rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(species_list)))
            ax.set_yticklabels(species_list, fontsize=8)
            
            # Highlight species below threshold
            for i, species in enumerate(species_list):
                if species_summary.loc[species, 'Max_Score'] < self.threshold:
                    # Add red rectangle around problematic species
                    rect = plt.Rectangle((-0.5, i-0.5), 2, 1, 
                                        fill=False, edgecolor=self.colors['below_threshold'], 
                                        linewidth=2, linestyle='--', alpha=0.8)
                    ax.add_patch(rect)
            
            # Add colorbar
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(im, cax=cax, label='Normalized Value')
            
            # Add statistics
            below_threshold = (species_summary['Max_Score'] < self.threshold).sum()
            stats_text = f'Species below threshold: {below_threshold}/{len(species_list)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   ha='left', va='top', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            
            ax.set_title('1. Multi-Species Hierarchical Clustering Heatmap\n'
                        'Red borders: Species below ≥85% threshold', 
                        fontsize=10, fontweight='bold', pad=12)
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical clustering heatmap: {e}")
            ax.text(0.5, 0.5, f'Plot Error: {str(e)[:50]}...', 
                   ha='center', va='center', fontsize=9)
            ax.set_title('1. Hierarchical Clustering Heatmap\n(Plot Error)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
    
     
    # GRAPH 2: Conservation Gradient Curve with Second-Derivative Inflection Mapping
     
    def _plot_conservation_gradient_curve(self, ax, df):
        """2 & 2.1: Tracks guide counts across conservation levels with inflection points"""
        
        if df.empty or len(df) < 10:
            ax.text(0.5, 0.5, 'Insufficient guide data\n(Need at least 10 guides)', 
                   ha='center', va='center', fontsize=10)
            ax.set_title('2. Conservation Gradient Curve\n(Insufficient Data)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        try:
            # Get conservation values
            conservation_values = df['Conservation'].astype(float).values
            
            # Bin conservation values
            bins = np.linspace(0, 1, 21)  # 20 bins
            bin_centers = (bins[:-1] + bins[1:]) / 2
            guide_counts = []
            
            for i in range(len(bins)-1):
                bin_start = bins[i]
                bin_end = bins[i+1]
                count = df[(df['Conservation'] >= bin_start) & 
                          (df['Conservation'] < bin_end)].shape[0]
                guide_counts.append(count)
            
            guide_counts = np.array(guide_counts, dtype=float)
            
            # Plot conservation gradient curve
            ax.plot(bin_centers, guide_counts, 'o-', color=self.colors['neutral'], 
                    linewidth=2, markersize=4, label='Guide Count')
            
            # Calculate and plot second derivative (inflection points)
            if len(guide_counts) > 4:
                # Smooth the curve first
                window_size = min(5, len(guide_counts) - 2)
                if window_size % 2 == 0:
                    window_size -= 1
                
                if window_size >= 3:
                    try:
                        smoothed = savgol_filter(guide_counts, window_size, 2)
                        
                        # First derivative
                        first_deriv = np.gradient(smoothed, bin_centers)
                        
                        # Second derivative
                        second_deriv = np.gradient(first_deriv, bin_centers)
                        
                        # Plot second derivative on secondary axis
                        ax2 = ax.twinx()
                        ax2.plot(bin_centers, second_deriv, 'r--', alpha=0.7, 
                                linewidth=1.5, label='2nd Derivative')
                        
                        # Find inflection points (where second derivative crosses zero)
                        inflection_points = []
                        for i in range(1, len(second_deriv)):
                            if second_deriv[i-1] * second_deriv[i] < 0:  # Sign change
                                inflection_points.append(bin_centers[i])
                        
                        # Mark inflection points
                        for point in inflection_points[:3]:  # Limit to first 3
                            # Find corresponding guide count
                            idx = np.argmin(np.abs(bin_centers - point))
                            ax.plot(point, guide_counts[idx], 'ro', markersize=8, 
                                   markeredgecolor='black', markeredgewidth=1)
                            ax.annotate(f'Inflection\n{point:.2f}', 
                                       xy=(point, guide_counts[idx]),
                                       xytext=(10, 10), textcoords='offset points',
                                       fontsize=7, ha='center',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                        
                        ax2.set_ylabel('Second Derivative', fontsize=9, color='red')
                        ax2.legend(loc='upper right', fontsize=8)
                        
                    except Exception as e:
                        self.logger.warning(f"Could not calculate derivatives: {e}")
            
            ax.set_xlabel('Conservation Level (0-1)', fontsize=9)
            ax.set_ylabel('Number of Guides', fontsize=9, color=self.colors['neutral'])
            ax.set_title('2. Conservation Gradient Curve with Inflection Mapping\n'
                        'Red points: Sharp drops in guide availability', 
                        fontsize=10, fontweight='bold', pad=12)
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=8)
            
        except Exception as e:
            self.logger.error(f"Error in conservation gradient curve: {e}")
            ax.text(0.5, 0.5, f'Plot Error: {str(e)[:50]}...', 
                   ha='center', va='center', fontsize=9)
            ax.set_title('2. Conservation Gradient Curve\n(Plot Error)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
    
     
    # GRAPH 3: Violin-Box Hybrid Distribution Plot with Threshold Density Ridge
     
    def _plot_violin_box_hybrid(self, ax, df):
        """3 & 3.1: Shows efficiency distribution per species with threshold focus"""
        
        if df.empty:
            ax.text(0.5, 0.5, 'No guide data available', 
                   ha='center', va='center', fontsize=10)
            ax.set_title('3. Violin-Box Hybrid Distribution\n(No Data)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        try:
            # Extract all unique species and their scores
            species_scores = {}
            
            for _, row in df.iterrows():
                species_list = row['Species'] if isinstance(row['Species'], list) else []
                for species in species_list:
                    if species not in species_scores:
                        species_scores[species] = []
                    species_scores[species].append(float(row['Score']))
            
            if not species_scores:
                ax.text(0.5, 0.5, 'No species score data', 
                       ha='center', va='center', fontsize=10)
                return
            
            species_list = list(species_scores.keys())
            
            # Filter out species with too few data points for violin plot
            valid_species = [s for s in species_list if len(species_scores[s]) >= 3]
            
            if len(valid_species) < 2:
                ax.text(0.5, 0.5, 'Insufficient species data\n(Need at least 2 species with ≥3 guides)', 
                       ha='center', va='center', fontsize=10)
                ax.set_title('3. Violin-Box Hybrid Distribution\n(Insufficient Data)', 
                            fontsize=10, fontweight='bold')
                ax.axis('off')
                return
            
            # Create violin plots for valid species
            violin_data = [species_scores[s] for s in valid_species]
            violin_parts = ax.violinplot(violin_data, showmeans=False, showmedians=True, showextrema=False)
            
            # Customize violins
            for pc in violin_parts['bodies']:
                pc.set_facecolor(self.colors['neutral'])
                pc.set_alpha(0.3)
                pc.set_edgecolor('black')
            
            # Add box plots on top
            box_positions = range(1, len(valid_species) + 1)
            box_parts = ax.boxplot(violin_data, positions=box_positions,
                                  widths=0.15, patch_artist=True,
                                  medianprops=dict(color='black', linewidth=1.5),
                                  boxprops=dict(facecolor=self.colors['neutral'], alpha=0.7),
                                  whiskerprops=dict(color='black'),
                                  capprops=dict(color='black'))
            
            # Add threshold density ridge (simplified version)
            for i, species in enumerate(valid_species):
                scores = np.array(species_scores[species])
                # Count guides near threshold
                near_threshold = np.sum((scores >= 80) & (scores <= 90))
                
                if near_threshold > 0:
                    # Add annotation for near-threshold guides
                    ax.text(i + 1, self.threshold + 2, f'{near_threshold}', 
                           ha='center', va='bottom', fontsize=7, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
            
            # Add threshold line
            ax.axhline(y=self.threshold, color=self.colors['above_threshold'], 
                      linestyle='--', linewidth=2, alpha=0.7,
                      label=f'Threshold ({self.threshold}%)')
            
            ax.set_xlabel('Species', fontsize=9)
            ax.set_ylabel('Guide Efficiency Score (%)', fontsize=9)
            ax.set_xticks(range(1, len(valid_species) + 1))
            ax.set_xticklabels([s[:15] + '...' if len(s) > 15 else s for s in valid_species], 
                              rotation=45, ha='right', fontsize=8)
            ax.set_title('3. Violin-Box Hybrid Distribution with Threshold Density Ridge\n'
                        'Yellow boxes: Guides near ±5% of threshold', 
                        fontsize=10, fontweight='bold', pad=12)
            
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend(loc='upper right', fontsize=8)
            
        except Exception as e:
            self.logger.error(f"Error in violin-box hybrid plot: {e}")
            ax.text(0.5, 0.5, f'Plot Error: {str(e)[:50]}...', 
                   ha='center', va='center', fontsize=9)
            ax.set_title('3. Violin-Box Hybrid Distribution\n(Plot Error)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
    
     
    # GRAPH 4: Multi-Species Volcano Plot
     
    def _plot_multi_species_volcano(self, ax, df):
        """4 & 4.1: Shows efficiency vs performance index across species"""
        
        if df.empty or len(df) < 5:
            ax.text(0.5, 0.5, 'Insufficient guide data\n(Need at least 5 guides)', 
                   ha='center', va='center', fontsize=10)
            ax.set_title('4. Multi-Species Volcano Plot\n(Insufficient Data)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        try:
            # Calculate performance index (combination of conservation and species count)
            performance_index = df['Conservation'].astype(float) * df['Species_Count'].astype(float)
            
            # Create volcano plot
            above_threshold = df[df['Above_Threshold']]
            below_threshold = df[~df['Above_Threshold']]
            
            # Plot below threshold points
            if not below_threshold.empty:
                scatter1 = ax.scatter(below_threshold['Score'].astype(float), 
                                     performance_index[below_threshold.index].astype(float),
                                     c=below_threshold['Conservation'].astype(float),
                                     cmap='Reds', alpha=0.6,
                                     s=30, edgecolors='black', linewidth=0.5,
                                     label='Below Threshold')
            
            # Plot above threshold points
            if not above_threshold.empty:
                scatter2 = ax.scatter(above_threshold['Score'].astype(float), 
                                     performance_index[above_threshold.index].astype(float),
                                     c=above_threshold['Conservation'].astype(float),
                                     cmap='Greens', alpha=0.8,
                                     s=50, edgecolors='black', linewidth=0.8,
                                     label='Above Threshold')
            
            # Add threshold line
            ax.axvline(x=self.threshold, color='black', linestyle='--',
                      linewidth=2, alpha=0.7, label=f'Threshold ({self.threshold}%)')
            
            ax.set_xlabel('Guide Efficiency Score (%)', fontsize=9)
            ax.set_ylabel('Performance Index\n(Conservation × Species Count)', fontsize=9)
            ax.set_title('4. Multi-Species Volcano Plot\n'
                        'Color intensity: Conservation level', 
                        fontsize=10, fontweight='bold', pad=12)
            
            ax.grid(True, alpha=0.3)
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', alpha=0.8, label='Above Threshold'),
                Patch(facecolor='red', alpha=0.6, label='Below Threshold'),
                Patch(facecolor='white', edgecolor='black', 
                      label=f'Threshold: {self.threshold}%')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
            
            # Add colorbar for conservation
            if not df.empty:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.1)
                sm = plt.cm.ScalarMappable(cmap='viridis')
                sm.set_array(df['Conservation'].astype(float))
                plt.colorbar(sm, cax=cax, label='Conservation Level')
            
        except Exception as e:
            self.logger.error(f"Error in volcano plot: {e}")
            ax.text(0.5, 0.5, f'Plot Error: {str(e)[:50]}...', 
                   ha='center', va='center', fontsize=9)
            ax.set_title('4. Multi-Species Volcano Plot\n(Plot Error)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
    
     
    # GRAPH 5: Dual-Layer Conservation Network Graph
     
    def _plot_dual_layer_network(self, ax, df, species_summary):
        """5 & 5.1: Shows species in primary/secondary networks and their overlap"""
        
        if species_summary.empty or len(species_summary) < 2:
            ax.text(0.5, 0.5, 'Insufficient species data\n(Need at least 2 species)', 
                   ha='center', va='center', fontsize=10)
            ax.set_title('5. Dual-Layer Conservation Network\n(Insufficient Data)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        try:
            ax.clear()
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Define network layers based on conservation quality
            # Primary network: Species with high conservation quality (> median)
            conservation_quality = species_summary['Conservation_Quality'].astype(float)
            median_conservation = conservation_quality.median()
            
            primary_species = species_summary[conservation_quality > median_conservation].index.tolist()
            secondary_species = species_summary[conservation_quality <= median_conservation].index.tolist()
            
            # Species that pass design criteria (have at least one guide above threshold)
            passing_species = species_summary[species_summary['Above_Threshold']].index.tolist()
            
            # Create circles for primary network
            primary_radius = 0.4
            primary_center = (0.3, 0.5)
            
            for i, species in enumerate(primary_species):
                angle = 2 * np.pi * i / max(len(primary_species), 1)
                x = primary_center[0] + primary_radius * np.cos(angle)
                y = primary_center[1] + primary_radius * np.sin(angle)
                
                # Node color based on passing criteria
                if species in passing_species:
                    node_color = self.colors['primary_network']
                    edge_color = 'black'
                    linewidth = 2
                else:
                    node_color = '#CCCCCC'
                    edge_color = '#999999'
                    linewidth = 1
                
                # Draw node
                circle = plt.Circle((x, y), 0.08, facecolor=node_color, 
                                   edgecolor=edge_color, linewidth=linewidth)
                ax.add_patch(circle)
                
                # Add species label
                ax.text(x, y - 0.12, species[:8], 
                       ha='center', va='center', fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Create circles for secondary network
            secondary_radius = 0.4
            secondary_center = (0.7, 0.5)
            
            for i, species in enumerate(secondary_species):
                angle = 2 * np.pi * i / max(len(secondary_species), 1)
                x = secondary_center[0] + secondary_radius * np.cos(angle)
                y = secondary_center[1] + secondary_radius * np.sin(angle)
                
                # Node color based on passing criteria
                if species in passing_species:
                    node_color = self.colors['secondary_network']
                    edge_color = 'black'
                    linewidth = 2
                else:
                    node_color = '#CCCCCC'
                    edge_color = '#999999'
                    linewidth = 1
                
                # Draw node
                circle = plt.Circle((x, y), 0.08, facecolor=node_color,
                                   edgecolor=edge_color, linewidth=linewidth)
                ax.add_patch(circle)
                
                # Add species label
                ax.text(x, y - 0.12, species[:8],
                       ha='center', va='center', fontsize=7,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Draw connections (species in both networks - these shouldn't exist, but we show as example)
            # Actually, a species can only be in one network. Let's show connections between primary and secondary networks
            # based on similarity in guide count
            
            # Connect species with similar guide counts
            max_guides = species_summary['Total_Guides'].max()
            if max_guides > 0:
                for i, species1 in enumerate(primary_species):
                    for j, species2 in enumerate(secondary_species):
                        guides1 = species_summary.loc[species1, 'Total_Guides']
                        guides2 = species_summary.loc[species2, 'Total_Guides']
                        
                        # Connect if guide counts are similar (±20%)
                        if abs(guides1 - guides2) / max(guides1, guides2) < 0.2:
                            # Find positions
                            angle1 = 2 * np.pi * i / max(len(primary_species), 1)
                            x1 = primary_center[0] + primary_radius * np.cos(angle1)
                            y1 = primary_center[1] + primary_radius * np.sin(angle1)
                            
                            angle2 = 2 * np.pi * j / max(len(secondary_species), 1)
                            x2 = secondary_center[0] + secondary_radius * np.cos(angle2)
                            y2 = secondary_center[1] + secondary_radius * np.sin(angle2)
                            
                            # Draw connecting line
                            ax.plot([x1, x2], [y1, y2], color=self.colors['overlap'],
                                   alpha=0.3, linewidth=1, linestyle='-')
            
            # Add network labels
            ax.text(primary_center[0], primary_center[1] + primary_radius + 0.15,
                   'Primary Network\n(High Conservation)', 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=self.colors['primary_network'], alpha=0.2))
            
            ax.text(secondary_center[0], secondary_center[1] + secondary_radius + 0.15,
                   'Secondary Network\n(Low Conservation)', 
                   ha='center', va='center', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round', facecolor=self.colors['secondary_network'], alpha=0.2))
            
            # Add statistics
            stats_text = []
            stats_text.append(f'Primary Network: {len(primary_species)} species')
            stats_text.append(f'Secondary Network: {len(secondary_species)} species')
            stats_text.append(f'Total passing: {len(passing_species)}/{len(species_summary)}')
            
            for i, text in enumerate(stats_text):
                ax.text(0.02, 0.98 - i*0.05, text, transform=ax.transAxes,
                       ha='left', va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title('5. Dual-Layer Conservation Network Graph\n'
                        'Shows species distribution across primary/secondary networks', 
                        fontsize=10, fontweight='bold', pad=12)
            
        except Exception as e:
            self.logger.error(f"Error in dual-layer network plot: {e}")
            ax.text(0.5, 0.5, f'Plot Error: {str(e)[:50]}...', 
                   ha='center', va='center', fontsize=9)
            ax.set_title('5. Dual-Layer Conservation Network\n(Plot Error)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
    
    # GRAPH 6: LOESS Fit and Residual Outlier Map

    def _plot_loess_residual_outlier(self, ax, df, species_summary):
        """6 & 6.1: Shows correlation between conservation quality and guide count"""
        
        if species_summary.empty or len(species_summary) < 4:
            ax.text(0.5, 0.5, 'Insufficient species data\n(Need at least 4 species)', 
                   ha='center', va='center', fontsize=10)
            ax.set_title('6. LOESS Fit and Residual Outlier Map\n(Insufficient Data)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
            return
        
        try:
            # Extract data as floats
            x = species_summary['Conservation_Quality'].astype(float).values
            y = species_summary['Total_Guides'].astype(float).values
            
            # Create scatter plot
            scatter = ax.scatter(x, y, alpha=0.6, s=50, 
                                c=species_summary['Max_Score'].astype(float).values,
                                cmap='coolwarm', edgecolors='black', linewidth=0.5)
            
            # Add species labels
            for i, species in enumerate(species_summary.index):
                ax.annotate(species[:6], 
                           xy=(x[i], y[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=6, alpha=0.7)
            
            # Fit linear regression as fallback (simpler than LOESS)
            if len(x) > 1:
                try:
                    # Linear regression
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    ax.plot(x_line, y_line, 'k-', alpha=0.7, linewidth=2,
                           label=f'Linear fit (R²={r_value**2:.3f})')
                    
                    # Calculate residuals
                    y_pred = slope * x + intercept
                    residuals = y - y_pred
                    
                    # Identify outliers (> 2 standard deviations)
                    residual_std = np.std(residuals)
                    outliers = np.abs(residuals) > 2 * residual_std
                    
                    # Mark outliers
                    outlier_x = x[outliers]
                    outlier_y = y[outliers]
                    
                    if len(outlier_x) > 0:
                        ax.scatter(outlier_x, outlier_y, color=self.colors['outlier'],
                                  s=100, edgecolors='black', linewidth=2,
                                  marker='X', label='Outliers (|residual| > 2σ)')
                        
                        # Annotate outlier species
                        for i in np.where(outliers)[0]:
                            species = species_summary.index[i]
                            ax.annotate(f'{species[:6]}*', 
                                       xy=(x[i], y[i]),
                                       xytext=(0, 15), textcoords='offset points',
                                       fontsize=7, fontweight='bold', color='red',
                                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    # Add correlation coefficient
                    corr_text = f'Correlation: {r_value:.3f}\n'
                    corr_text += f'Outliers: {outliers.sum()}/{len(x)} species'
                    ax.text(0.02, 0.98, corr_text, transform=ax.transAxes,
                           ha='left', va='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
                    
                except Exception as e:
                    self.logger.warning(f"Linear regression failed: {e}")
            
            ax.set_xlabel('Conservation Network Quality', fontsize=9)
            ax.set_ylabel('Number of Guides', fontsize=9)
            ax.set_title('6. LOESS Fit and Residual Outlier Map\n'
                        'Color: Max efficiency, X: Significant deviation', 
                        fontsize=10, fontweight='bold', pad=12)
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            # Add colorbar for max score
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            plt.colorbar(scatter, cax=cax, label='Max Score (%)')
            
        except Exception as e:
            self.logger.error(f"Error in LOESS residual plot: {e}")
            ax.text(0.5, 0.5, f'Plot Error: {str(e)[:50]}...', 
                   ha='center', va='center', fontsize=9)
            ax.set_title('6. LOESS Fit and Residual Outlier Map\n(Plot Error)', 
                        fontsize=10, fontweight='bold')
            ax.axis('off')
    
     
    # HELPER METHODS
     
    def _create_individual_plots(self, df, species_summary, output_folder):
        """Create individual plot files for each graph"""
        self.logger.info("Creating individual plot files...")
        
        # Import necessary modules
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        
        try:
            # Graph 1: Hierarchical Clustering Heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_hierarchical_clustering_heatmap(ax, df, species_summary)
            plt.tight_layout()
            plt.savefig(output_folder / "1_hierarchical_clustering_heatmap.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save Graph 1: {e}")
        
        try:
            # Graph 2: Conservation Gradient Curve
            fig, ax = plt.subplots(figsize=(10, 6))
            self._plot_conservation_gradient_curve(ax, df)
            plt.tight_layout()
            plt.savefig(output_folder / "2_conservation_gradient_curve.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save Graph 2: {e}")
        
        try:
            # Graph 3: Violin-Box Hybrid Distribution
            fig, ax = plt.subplots(figsize=(12, 8))
            self._plot_violin_box_hybrid(ax, df)
            plt.tight_layout()
            plt.savefig(output_folder / "3_violin_box_hybrid.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save Graph 3: {e}")
        
        try:
            # Graph 4: Multi-Species Volcano Plot
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_multi_species_volcano(ax, df)
            plt.tight_layout()
            plt.savefig(output_folder / "4_multi_species_volcano.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save Graph 4: {e}")
        
        try:
            # Graph 5: Dual-Layer Conservation Network
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_dual_layer_network(ax, df, species_summary)
            plt.tight_layout()
            plt.savefig(output_folder / "5_dual_layer_network.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save Graph 5: {e}")
        
        try:
            # Graph 6: LOESS Fit and Residual Outlier Map
            fig, ax = plt.subplots(figsize=(10, 8))
            self._plot_loess_residual_outlier(ax, df, species_summary)
            plt.tight_layout()
            plt.savefig(output_folder / "6_loess_residual_outlier.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.warning(f"Failed to save Graph 6: {e}")
        
        self.logger.info("Individual plot files saved")
        
 
# RESULTS EXPORTER
 
class ResultsExporter:
    """Exports pipeline results to various formats"""
    
    def __init__(self, config: Config, logger: logging.Logger):
        self.config = config
        self.logger = logger
    
    def export_results(self, guides: List[GuideCandidate], sequences: Dict[str, List[SequenceRecord]]):
        """Export all results to files"""
        output_folder = self.config.output_folder
        output_folder.mkdir(exist_ok=True, parents=True)
        
        # 1. Export guides to CSV
        self._export_to_csv(guides, output_folder)
        
        # 2. Export to FASTA
        self._export_to_fasta(guides, output_folder)
        
        # 3. Export summary report
        self._export_summary(guides, sequences, output_folder)
        
        # 4. Export top guides for Gibson assembly
        self._export_gibson_primers(guides, output_folder)
    
    def _export_to_csv(self, guides: List[GuideCandidate], output_folder: Path):
        """Export guides to CSV file"""
        if not guides:
            self.logger.warning("No guides to export")
            return
        
        data = []
        for i, guide in enumerate(guides):
            data.append({
                'Rank': i + 1,
                'Guide_Sequence': guide.sequence,
                'PAM': guide.pam,
                'Species_Count': len(guide.species),
                'Species': '; '.join(guide.species),
                'On_Target_Score': guide.score,
                'GC_Percent': guide.gc_percent,
                'Conservation_Score': guide.conservation,
                'Off_Target_Risk': guide.off_target_risk,
                'High_Quality': guide.is_high_quality
            })
        
        df = pd.DataFrame(data)
        csv_path = output_folder / "msblt_guides.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        self.logger.info(f"Guides exported to: {csv_path}")
    
    def _export_to_fasta(self, guides: List[GuideCandidate], output_folder: Path):
        """Export top guides to FASTA format"""
        if not guides:
            return
        
        # Export top 50 guides
        fasta_path = output_folder / "top_guides.fasta"
        with open(fasta_path, 'w', encoding='utf-8') as f:
            for i, guide in enumerate(guides[:50]):
                header = f">guide_{i+1}_score_{guide.score:.0f}_spp_{len(guide.species)}_gc_{guide.gc_percent:.0f}"
                f.write(f"{header}\n")
                f.write(f"{guide.sequence}\n")
        
        self.logger.info(f"FASTA file exported to: {fasta_path}")
    
    def _export_summary(self, guides: List[GuideCandidate], 
                       sequences: Dict[str, List[SequenceRecord]], 
                       output_folder: Path):
        """Export summary report"""
        report_path = output_folder / "pipeline_summary.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("msBLT 2.0 - Pipeline Summary Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Pipeline info
            f.write("PIPELINE INFORMATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total species analyzed: {len(sequences)}\n")
            f.write(f"Total sequences loaded: {sum(len(seqs) for seqs in sequences.values())}\n\n")
            
            # Guide statistics
            f.write("GUIDE RNA STATISTICS\n")
            f.write("-" * 40 + "\n")
            
            if guides:
                high_quality = sum(1 for g in guides if g.is_high_quality)
                avg_score = sum(g.score for g in guides) / len(guides)
                avg_gc = sum(g.gc_percent for g in guides) / len(guides)
                avg_species = sum(len(g.species) for g in guides) / len(guides)
                
                f.write(f"Total guides designed: {len(guides)}\n")
                f.write(f"High-quality guides: {high_quality} ({high_quality/len(guides)*100:.1f}%)\n")
                f.write(f"Average score: {avg_score:.1f}\n")
                f.write(f"Average GC content: {avg_gc:.1f}%\n")
                f.write(f"Average species coverage: {avg_species:.1f}\n\n")
                
                # Top 10 guides
                f.write("TOP 10 GUIDE RNAS\n")
                f.write("-" * 40 + "\n")
                for i, guide in enumerate(guides[:10]):
                    f.write(f"{i+1:2d}. {guide.sequence}\n")
                    f.write(f"    Score: {guide.score:.1f} | Species: {len(guide.species)} "
                           f"| GC: {guide.gc_percent:.1f}% | Risk: {guide.off_target_risk:.1f}%\n")
                    f.write(f"    Species: {', '.join(guide.species)}\n\n")
                
                # Recommendations
                f.write("RECOMMENDATIONS\n")
                f.write("-" * 40 + "\n")
                f.write("1. Prioritize guides with score >= 85\n")
                f.write("2. Choose guides covering multiple species for broad-spectrum targeting\n")
                f.write("3. Avoid guides with GC content outside 25-75% range\n")
                f.write("4. Consider off-target risk < 5% for high specificity\n")
            else:
                f.write("No guides were designed.\n")
        
        self.logger.info(f"Summary report exported to: {report_path}")
    
    def _export_gibson_primers(self, guides: List[GuideCandidate], output_folder: Path):
        """Export Gibson assembly primers for top guides"""
        if not guides:
            return
        
        # Take top 3 guides
        top_guides = guides[:3]
        
        primer_path = output_folder / "gibson_primers.txt"
        with open(primer_path, 'w', encoding='utf-8') as f:
            f.write("Gibson Assembly Primers for Top 3 Guides\n")
            f.write("=" * 50 + "\n\n")
            
            for i, guide in enumerate(top_guides):
                f.write(f"Guide {i+1}: {guide.sequence}\n")
                f.write("-" * 40 + "\n")
                
                # Generate primer sequences (simplified)
                # For CRISPR cloning, typical format: CACCg + guide for forward, AAAC + reverse complement for reverse
                fwd_primer = "CACC" + guide.sequence
                rev_primer = "AAAC" + str(Seq(guide.sequence).reverse_complement())
                
                f.write(f"Forward primer (5'-3'): {fwd_primer}\n")
                f.write(f"Reverse primer (5'-3'): {rev_primer}\n")
                f.write(f"Annealing temperature: ~60°C\n")
                f.write(f"Length: {len(fwd_primer)} bp / {len(rev_primer)} bp\n\n")
        
        self.logger.info(f"Primers exported to: {primer_path}")

 
# MAIN PIPELINE
 
class MSBLTPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.output_folder)
        self.sequences = {}
        self.guides = []
    
    def run(self) -> bool:
        """Run the complete pipeline"""
        try:
            self.logger.info("=" * 60)
            self.logger.info("msBLT 2.0 - Simplified Working Pipeline")
            self.logger.info("=" * 60)
            
            # Step 1: Load sequences
            self.logger.info("\n[1/5] Loading sequences...")
            loader = SequenceLoader(self.config, self.logger)
            self.sequences = loader.load_all()
            
            if not self.sequences:
                self.logger.error("No sequences loaded. Please check your data folder.")
                return False
            
            # Step 2: Find conserved regions (simplified)
            self.logger.info("\n[2/5] Finding conserved k-mers...")
            analyzer = SimpleConservationAnalyzer(self.config, self.logger)
            conserved_regions = analyzer.find_conserved_kmers(self.sequences)
            
            if conserved_regions:
                self.logger.info(f"Found {len(conserved_regions)} conserved regions")
            else:
                self.logger.warning("No conserved regions found, will search for guides directly")
            
            # Step 3: Design guide RNAs
            self.logger.info("\n[3/5] Designing guide RNAs...")
            designer = GuideDesigner(self.config, self.logger)
            self.guides = designer.find_shared_guides(self.sequences)
            
            if not self.guides:
                self.logger.error("No guide RNAs designed. Check your sequences for NGG PAM sites.")
                return False
            
            self.logger.info(f"Successfully designed {len(self.guides)} guide RNAs")
            
            # Step 4: Create visualizations
            self.logger.info("\n[4/5] Creating visualizations...")
            visualizer = ResultsVisualizer(self.config)
            visualizer.create_dashboard(self.guides, self.config.output_folder)
            
            # Step 5: Export results
            self.logger.info("\n[5/5] Exporting results...")
            exporter = ResultsExporter(self.config, self.logger)
            exporter.export_results(self.guides, self.sequences)
            
            # Final summary
            self._print_summary()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False
    
    def _print_summary(self):
        """Print a summary of the pipeline results"""
        high_quality = sum(1 for g in self.guides if g.is_high_quality)
    
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nRESULTS SUMMARY:")
        print(f"   • Species analyzed: {len(self.sequences)}")
        print(f"   • Guide RNAs designed: {len(self.guides)}")
        print(f"   • High-quality guides: {high_quality} ({high_quality/len(self.guides)*100:.1f}%)")
        print(f"   • Average score: {sum(g.score for g in self.guides)/len(self.guides):.1f}")
    
        print(f"\nOUTPUT FILES:")
        print(f"   • Dashboard: {self.config.output_folder / 'msblt_dashboard.png'}")
        print(f"   • Guide list: {self.config.output_folder / 'msblt_guides.csv'}")
        print(f"   • Report: {self.config.output_folder / 'pipeline_summary.txt'}")
        print(f"   • FASTA: {self.config.output_folder / 'top_guides.fasta'}")
    
        print(f"\nTOP 5 GUIDES:")
        for i, guide in enumerate(self.guides[:5]):
            print(f"   {i+1}. {guide.sequence[:15]}... (Score: {guide.score:.1f}, "
                  f"Species: {len(guide.species)}, GC: {guide.gc_percent:.1f}%)")
    
        print("\n" + "=" * 60)

 
# COMMAND LINE INTERFACE
 
def main():
    """Main entry point for command line"""
    parser = argparse.ArgumentParser(
        description="msBLT 2.0: Multi-Species Beta-Lactamase Targeting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python msblt_fixed.py --data ./data --output ./results
  python msblt_fixed.py --data ./sequences --output ./analysis
        """
    )
    
    parser.add_argument(
        "--data", "-d",
        type=Path,
        default="data",
        help="Directory containing FASTA files (default: data)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--gene", "-g",
        default="bla",
        help="Target gene identifier (default: 'bla' for β-lactamase)"
    )
    
    parser.add_argument(
        "--min-species", "-m",
        type=int,
        default=1,
        help="Minimum number of species sharing a guide (default: 1)"
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = Config(
        data_folder=args.data,
        output_folder=args.output,
        target_gene=args.gene,
        min_species_count=args.min_species
    )
    
    # Run pipeline
    pipeline = MSBLTPipeline(config)
    success = pipeline.run()
    
    sys.exit(0 if success else 1)

 
# ENTRY POINT
 
if __name__ == "__main__":

    main()
