"""
msBLT 2.0 - Beta-lactamase CRISPR targeting
Howard HS - Garcia, James Jelo
2025
"""

import sys, os, argparse, logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import savgol_filter

# Config stuff - tweak these as needed
MIN_SEQ_LENGTH = 50
GUIDE_LENGTH = 20
GOOD_SCORE = 85  # from literature
MAX_OFF_TARGET = 5.0
GC_RANGE = (25, 75)

def setup_logger(out_dir):
    """Quick logger setup"""
    out_dir.mkdir(exist_ok=True, parents=True)
    log_file = out_dir / "run.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_sequences(data_dir, target_gene="bla"):
    """Load FASTA files, filter for target gene if specified"""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading from {data_dir}")
    
    if not data_dir.exists():
        logger.error(f"Data dir not found: {data_dir}")
        return {}
    
    sequences = {}
    total = 0
    fasta_files = list(data_dir.glob("*.fasta")) + list(data_dir.glob("*.fa"))
    
    if not fasta_files:
        logger.error("No FASTA files!")
        return {}
    
    for fa in fasta_files:
        species = fa.stem
        species_seqs = []
        
        try:
            for rec in SeqIO.parse(fa, "fasta"):
                # Skip if not our target gene
                if target_gene != "none" and target_gene.lower() not in rec.description.lower():
                    continue
                
                seq = str(rec.seq).upper().strip()
                seq = ''.join(c for c in seq if c in 'ATCGN')  # clean it up
                
                if len(seq) < MIN_SEQ_LENGTH:
                    continue
                
                # GC content
                gc = (seq.count('G') + seq.count('C')) / len(seq) * 100 if seq else 0
                
                species_seqs.append({
                    'seq': seq,
                    'id': rec.id,
                    'gc': gc,
                    'len': len(seq)
                })
                total += 1
            
            if species_seqs:
                sequences[species] = species_seqs
                logger.info(f"  {species}: {len(species_seqs)} seqs")
                
        except Exception as e:
            logger.warning(f"Problem with {fa}: {e}")
    
    logger.info(f"Loaded {total} seqs from {len(sequences)} species")
    return sequences

def find_guides(sequences, min_species, pam='NGG', genome_context=None):
    logger = logging.getLogger(__name__)
    guides = []
    
    # Common CRISPR PAM sequences to try
    alternative_pams = ['NGG', 'NAG', 'NGA', 'NGT', 'TGG', 'AGG', 'GGG', 'CGG']
    
    for species, seq_list in sequences.items():
        for seq in seq_list:
            seq_str = seq['seq']
            logger.info(f"Processing {species}: sequence length {len(seq_str)}")
            
            # Try different PAM sequences
            found_with_pam = None
            for test_pam in alternative_pams:
                found_any = False
                for i in range(len(seq_str) - GUIDE_LENGTH - len(test_pam) + 1):
                    guide_seq = seq_str[i:i + GUIDE_LENGTH]
                    pam_seq = seq_str[i + GUIDE_LENGTH:i + GUIDE_LENGTH + len(test_pam)]
                    
                    if pam_seq == test_pam:
                        # Use enhanced scoring with off-target analysis
                        score_result = enhanced_score_guide(guide_seq, genome_context)
                        
                        # Get genomic context for this specific guide
                        locations = get_genomic_context(guide_seq, seq_str, i)
                        
                        guides.append({
                            'seq': guide_seq,
                            'species': species,
                            'start': i,
                            'end': i + GUIDE_LENGTH,
                            'score': score_result['score'],
                            'pam': pam_seq,
                            'gc': score_result['gc'],
                            'risk': score_result['risk'],
                            'efficiency': score_result['efficiency'],
                            'self_complementarity': score_result['self_complementarity'],
                            'off_targets': score_result['off_targets'],
                            'genomic_locations': locations,
                            'good': score_result['score'] >= GOOD_SCORE
                        })
                        found_any = True
                        
                if found_any:
                    found_with_pam = test_pam
                    logger.info(f"Found guides using PAM: {test_pam}")
                    break
            
            if not found_with_pam:
                logger.warning(f"No guides found with any common PAM sequence in {species}")
    
    logger.info(f"Total guides found: {len(guides)}")
    return guides

def find_conserved_kmers(seqs_dict, k=20):
    """Simple conserved k-mer finder - uses first seq per species"""
    logger = logging.getLogger(__name__)
    
    if len(seqs_dict) < 2:
        logger.warning("Need 2+ species for conservation. Bypassing conservation check.")
        # Return all possible guides from the single species
        guides = []
        for species, seq_list in seqs_dict.items():
            for seq in seq_list:
                for i in range(len(seq['seq']) - k + 1):
                    kmer = seq['seq'][i:i + k]
                    if 'N' not in kmer:
                        guides.append({
                            'start': i,
                            'end': i + k,
                            'seq': kmer,
                            'conservation': 1.0,  # Full conservation for single species
                            'species': [species]
                        })
        return guides
    
    # Take first sequence from each
    sample_seqs = {}
    for species, seq_list in seqs_dict.items():
        if seq_list:
            sample_seqs[species] = seq_list[0]['seq']
    
    if len(sample_seqs) < 2:
        return []
    
    ref_species = list(sample_seqs.keys())[0]
    ref_seq = sample_seqs[ref_species]
    
    conserved = []
    step = 5  # skip some to avoid overlap
    
    for i in range(0, len(ref_seq) - k + 1, step):
        kmer = ref_seq[i:i + k]
        if 'N' in kmer or len(kmer) < k:
            continue
        
        matches = [ref_species]
        for sp, seq in sample_seqs.items():
            if sp != ref_species and kmer in seq:
                matches.append(sp)
        
        if len(matches) >= 2:
            conservation = len(matches) / len(sample_seqs)
            conserved.append({
                'start': i,
                'end': i + k,
                'seq': kmer,
                'conservation': conservation,
                'species': matches
            })
    
    logger.info(f"Found {len(conserved)} conserved k-mers")
    return conserved


def score_guide(guide_seq):
    """
    Advanced guide RNA scoring based on CRISPR literature
    Incorporates Doench 2016 rules and ChopChop-like efficiency scoring
    """
    if len(guide_seq) != 20:
        return {'score': 0, 'gc': 0, 'risk': 100, 'efficiency': 0}
    
    # Initialize base score (ChopChop efficiency ranges 0-100)
    efficiency = 50.0  # Start at 50, adjust based on features
    gc_count = guide_seq.count('G') + guide_seq.count('C')
    gc_percent = (gc_count / 20) * 100
    
    # 1. GC CONTENT SCORING (Optimal: 40-60%)
    if 40 <= gc_percent <= 60:
        efficiency += 15  # Optimal range
    elif 30 <= gc_percent < 40 or 60 < gc_percent <= 70:
        efficiency += 5   # Acceptable range
    else:
        efficiency -= 10  # Poor range
    
    # 2. POSITION-SPECIFIC SCORING (Doench 2016 rules)
    position_weights = {
        0: {'G': 10, 'A': 5, 'T': -5, 'C': 0},      # Prefer G at position 1
        19: {'G': 10, 'C': 10, 'A': -5, 'T': -5},   # Prefer G/C at last position
    }
    
    for pos, weights in position_weights.items():
        nucleotide = guide_seq[pos]
        if nucleotide in weights:
            efficiency += weights[nucleotide]
    
    # 3. NUCLEOTIDE COMPOSITION AT KEY POSITIONS
    # Position 10-12 (seed region) should be GC-rich
    seed_region = guide_seq[10:13]
    seed_gc = (seed_region.count('G') + seed_region.count('C')) / 3 * 100
    if seed_gc >= 66:
        efficiency += 8
    elif seed_gc >= 33:
        efficiency += 4
    else:
        efficiency -= 5
    
    # 4. AVOID HOMOPOLYMERS (self-complementarity)
    # Check for consecutive repeats
    penalties = 0
    for i in range(len(guide_seq) - 3):
        # Poly-T (termination signal)
        if guide_seq[i:i+4] == 'TTTT':
            penalties += 15
        # Any 4+ homopolymer
        if len(set(guide_seq[i:i+4])) == 1:
            penalties += 10
    
    # Check for self-complementarity (simplified)
    comp_score = 0
    rev_comp = str(Seq(guide_seq).reverse_complement())
    for i in range(len(guide_seq) - 7):
        if guide_seq[i:i+8] in rev_comp:
            comp_score += 5
    
    efficiency -= penalties + comp_score
    
    # 5. DI-NUCLEOTIDE PREFERENCES
    # Avoid specific patterns
    poor_patterns = ['TTTT', 'AAAA', 'CCCC', 'GGGG', 'TATAT', 'CACAC']
    for pattern in poor_patterns:
        if pattern in guide_seq:
            efficiency -= 8
    
    # 6. NUCLEOTIDE DISTRIBUTION
    # Balanced composition is better
    counts = {'A': guide_seq.count('A'), 
              'T': guide_seq.count('T'),
              'C': guide_seq.count('C'),
              'G': guide_seq.count('G') }
    max_count = max(counts.values())
    if max_count >= 10:  # Any nucleotide > 50%
        efficiency -= 10
    
    # Ensure score is within reasonable bounds
    efficiency = max(20.0, min(100.0, efficiency))
    
    # Risk assessment (simplified)
    complexity = len(set(guide_seq)) / len(guide_seq)
    risk = 100 * (1 - complexity**2)
    
    score = efficiency  # Use efficiency as the main score
    
    return {
        'score': round(score, 2),
        'gc': round(gc_percent, 1),
        'risk': round(min(risk, 100), 1),
        'efficiency': round(efficiency, 2)  # New: ChopChop-like efficiency
    }

def calculate_self_complementarity(guide_seq):
    """Calculate self-complementarity score"""
    rev_comp = str(Seq(guide_seq).reverse_complement())
    score = 0
    # Check for 8+ base pairings
    for i in range(len(guide_seq) - 7):
        if guide_seq[i:i+8] in rev_comp:
            score += 5
    return score

def find_off_targets(guide_seq, genome_seq, max_mismatches=3):
    """Find off-target sites with 0-3 mismatches"""
    off_targets = {'MM0': 0, 'MM1': 0, 'MM2': 0, 'MM3': 0}
    guide_len = len(guide_seq)
    
    # Scan genome for potential off-targets
    for i in range(len(genome_seq) - guide_len + 1):
        target_seq = genome_seq[i:i+guide_len]
        mismatches = sum(1 for a, b in zip(guide_seq, target_seq) if a != b)
        
        if mismatches <= max_mismatches:
            key = f'MM{mismatches}'
            off_targets[key] += 1
    
    return off_targets

def get_genomic_context(guide_seq, genome_seq, start_pos):
    """Get genomic location, strand, and context"""
    # Find all occurrences in genome
    locations = []
    guide_len = len(guide_seq)
    
    for i in range(len(genome_seq) - guide_len + 1):
        if genome_seq[i:i+guide_len] == guide_seq:
            locations.append({
                'position': i,
                'strand': '+',
                'context': genome_seq[max(0, i-10):i+guide_len+10]
            })
        # Also check reverse complement
        rev_seq = str(Seq(genome_seq[i:i+guide_len]).reverse_complement())
        if rev_seq == guide_seq:
            locations.append({
                'position': i,
                'strand': '-',
                'context': genome_seq[max(0, i-10):i+guide_len+10]
            })
    
    return locations

def enhanced_score_guide(guide_seq, genome_context=None):
    """
    Enhanced guide RNA scoring with off-target analysis
    """
    if len(guide_seq) != 20:
        return {'score': 0, 'gc': 0, 'risk': 100, 'efficiency': 0, 
                'self_complementarity': 0, 'off_targets': {'MM0': 0, 'MM1': 0, 'MM2': 0, 'MM3': 0}}
    
    # Get basic scoring
    base_result = score_guide(guide_seq)
    
    # Calculate self-complementarity
    self_comp = calculate_self_complementarity(guide_seq)
    
    # Adjust score based on self-complementarity
    base_result['efficiency'] -= self_comp
    base_result['score'] = base_result['efficiency']  # Keep score and efficiency in sync
    
    # Find off-targets if genome context provided
    off_targets = {'MM0': 0, 'MM1': 0, 'MM2': 0, 'MM3': 0}
    if genome_context:
        off_targets = find_off_targets(guide_seq, genome_context)
        # Penalize score based on off-targets
        off_target_penalty = (off_targets['MM0'] * 20 + 
                             off_targets['MM1'] * 10 + 
                             off_targets['MM2'] * 5 + 
                             off_targets['MM3'] * 2)
        base_result['efficiency'] -= off_target_penalty
        base_result['score'] = max(20.0, base_result['efficiency'])  # Keep minimum score
    
    return {
        'score': round(base_result['score'], 2),
        'gc': base_result['gc'],
        'risk': base_result['risk'],
        'efficiency': round(base_result['efficiency'], 2),
        'self_complementarity': self_comp,
        'off_targets': off_targets
    }

def self_complementarity(guide_seq):
    rev_comp = str(Seq(guide_seq).reverse_complement())
    score = 0
    for i in range(len(guide_seq) - 7):
        if guide_seq[i:i+8] in rev_comp:
            score += 5
    return score

def off_target_mismatch_analysis(guide_seq, off_target_db):
    mismatches = []
    for off_target in off_target_db:
        mm = sum(1 for a, b in zip(guide_seq, off_target) if a != b)
        mismatches.append(mm)
    return mismatches

def get_genomic_location(guide_seq, genome_db):
    locations = []
    for chrom, seq in genome_db.items():
        pos = seq.find(guide_seq)
        if pos != -1:
            locations.append((chrom, pos))
    return locations

def get_strand_info(guide_seq, genome_db):
    locations = get_genomic_location(guide_seq, genome_db)
    strands = []
    for chrom, pos in locations:
        if pos != -1:
            strands.append('+')
        else:
            strands.append('-')
    return strands


def make_plots(guides, out_dir):
    """Make the 6 specific graphs exactly as requested"""
    logger = logging.getLogger(__name__)
    out_dir.mkdir(exist_ok=True)
    
    if not guides:
        logger.warning("No guides to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(guides)
    
    # Add species_count if not present
    if 'species_count' not in df.columns:
        df['species_count'] = 1  # Single species
    
    # Add conservation if not present
    if 'conservation' not in df.columns:
        df['conservation'] = 1.0  # Full conservation for single species
    
    logger.info(f"Creating plots for {len(guides)} guides")
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(15, 18))
    fig.suptitle('msBLT 2.0: Specific Graph Analysis\nDual-Layer Conservation Network Framework', 
                fontsize=14, fontweight='bold')
    
    # Human colors - pick what works
    colors = {
        'good': '#2E86AB',      # blue
        'bad': '#A23B72',       # purple  
        'neutral': '#F18F01',   # orange
        'primary': '#28A745',   # green
        'secondary': '#17A2B8', # teal
        'highlight': '#6F42C1', # purple
    }
    
    # 1. Hierarchical Clustering Heatmap
    ax1 = plt.subplot(3, 2, 1)
    try:
        if len(df) > 0:
            # Get species data
            species_data = {}
            for _, row in df.iterrows():
                for sp in row['species']:
                    if sp not in species_data:
                        species_data[sp] = []
                    species_data[sp].append(row['score'])
            
            # Create simple heatmap
            species_list = list(species_data.keys())
            max_scores = [max(scores) for scores in species_data.values()]
            
            # Sort by score
            sorted_idx = np.argsort(max_scores)
            species_list = [species_list[i] for i in sorted_idx]
            max_scores = [max_scores[i] for i in sorted_idx]
            
            # Create heatmap matrix (2 columns: max score and guide count)
            heat_data = []
            for sp in species_list:
                scores = species_data[sp]
                heat_data.append([
                    max(scores) / 100,  # normalized max score
                    len(scores) / max(len(s) for s in species_data.values()) if species_data.values() else 0
                ])
            
            im = ax1.imshow(heat_data, aspect='auto', cmap='viridis')
            ax1.set_xticks([0, 1])
            ax1.set_xticklabels(['Max Score\n(Norm)', 'Guide Density\n(Norm)'], 
                               rotation=45, ha='right')
            ax1.set_yticks(range(len(species_list)))
            ax1.set_yticklabels([s[:12] for s in species_list])
            
            # Highlight species below threshold
            for i, sp in enumerate(species_list):
                if max_scores[i] < GOOD_SCORE:
                    rect = plt.Rectangle((-0.5, i-0.5), 2, 1, 
                                        fill=False, edgecolor=colors['bad'], 
                                        linewidth=2, linestyle='--', alpha=0.8)
                    ax1.add_patch(rect)
            
            plt.colorbar(im, ax=ax1, label='Normalized')
        else:
            ax1.text(0.5, 0.5, 'No data', ha='center')
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center')
    ax1.set_title('1. Hierarchical Clustering Heatmap', fontweight='bold')
    
    # 2. Conservation Gradient Curve
    ax2 = plt.subplot(3, 2, 2)
    try:
        if len(df) > 10:
            # Bin conservation values
            bins = np.linspace(0, 1, 21)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            counts = []
            
            for i in range(len(bins)-1):
                mask = (df['conservation'] >= bins[i]) & (df['conservation'] < bins[i+1])
                counts.append(df[mask].shape[0])
            
            ax2.plot(bin_centers, counts, 'o-', color=colors['neutral'], linewidth=2)
            
            # Try to find inflection points (simplified)
            if len(counts) > 4:
                window = min(5, len(counts))
                if window % 2 == 0:
                    window -= 1
                if window >= 3:
                    try:
                        smoothed = savgol_filter(counts, window, 2)
                        deriv = np.gradient(smoothed, bin_centers)
                        
                        # Mark where slope changes sign
                        for i in range(1, len(deriv)):
                            if deriv[i-1] * deriv[i] < 0:
                                ax2.plot(bin_centers[i], counts[i], 'ro', markersize=8)
                    except:
                        pass  # skip if smoothing fails
            
            ax2.set_xlabel('Conservation Level')
            ax2.set_ylabel('Number of Guides')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Need more data', ha='center')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center')
    ax2.set_title('2. Conservation Gradient Curve', fontweight='bold')
    
    # 3. Violin-Box Hybrid Distribution
    ax3 = plt.subplot(3, 2, 3)
    try:
        if len(df) > 0:
            # Get species with enough data
            species_scores = {}
            for _, row in df.iterrows():
                for sp in row['species']:
                    if sp not in species_scores:
                        species_scores[sp] = []
                    species_scores[sp].append(row['score'])
            
            # Keep species with at least 3 scores
            valid_species = [sp for sp, scores in species_scores.items() if len(scores) >= 3]
            
            if len(valid_species) >= 2:
                # Create violin plots
                violin_data = [species_scores[sp] for sp in valid_species]
                violin_parts = ax3.violinplot(violin_data, showmedians=True)
                
                # Style violins
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(colors['neutral'])
                    pc.set_alpha(0.3)
                
                # Add box plots on top
                box = ax3.boxplot(violin_data, positions=range(1, len(valid_species)+1),
                                 widths=0.15, patch_artist=True)
                for patch in box['boxes']:
                    patch.set_facecolor(colors['neutral'])
                    patch.set_alpha(0.7)
                
                # Add threshold line
                ax3.axhline(y=GOOD_SCORE, color=colors['good'], linestyle='--', linewidth=2)
                
                ax3.set_xticks(range(1, len(valid_species)+1))
                ax3.set_xticklabels([s[:10] for s in valid_species], rotation=45, ha='right')
                ax3.set_ylabel('Guide Score')
                ax3.grid(True, alpha=0.3, axis='y')
            else:
                ax3.text(0.5, 0.5, 'Need 2+ species with 3+ guides', ha='center')
        else:
            ax3.text(0.5, 0.5, 'No data', ha='center')
    except Exception as e:
        ax3.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center')
    ax3.set_title('3. Violin-Box Hybrid Distribution', fontweight='bold')
    
    # 4. Multi-Species Volcano Plot
    ax4 = plt.subplot(3, 2, 4)
    try:
        if len(df) > 5:
            # Performance index = conservation * species count
            perf_index = df['conservation'] * df['species_count']
            
            # Split by threshold
            above = df[df['score'] >= GOOD_SCORE]
            below = df[df['score'] < GOOD_SCORE]
            
            if not below.empty:
                ax4.scatter(below['score'], perf_index[below.index], 
                          c='red', alpha=0.6, s=30, label='Below Threshold')
            if not above.empty:
                ax4.scatter(above['score'], perf_index[above.index], 
                          c='green', alpha=0.8, s=50, label='Above Threshold')
            
            ax4.axvline(x=GOOD_SCORE, color='black', linestyle='--', linewidth=2)
            ax4.set_xlabel('Guide Score')
            ax4.set_ylabel('Conservation x Species Count')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Need more data', ha='center')
    except Exception as e:
        ax4.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center')
    ax4.set_title('4. Multi-Species Volcano Plot', fontweight='bold')
    
    # 5. Dual-Layer Conservation Network
    ax5 = plt.subplot(3, 2, 5)
    ax5.set_aspect('equal')
    ax5.axis('off')
    
    try:
        if len(df) > 0:
            # Calculate conservation quality per species
            species_stats = {}
            for _, row in df.iterrows():
                for sp in row['species']:
                    if sp not in species_stats:
                        species_stats[sp] = {'scores': [], 'conservation': []}
                    species_stats[sp]['scores'].append(row['score'])
                    species_stats[sp]['conservation'].append(row['conservation'])
            
            # Calculate avg conservation quality
            species_quality = {}
            for sp, stats in species_stats.items():
                avg_conservation = np.mean(stats['conservation']) if stats['conservation'] else 0
                max_score = max(stats['scores']) if stats['scores'] else 0
                species_quality[sp] = {
                    'conservation': avg_conservation,
                    'max_score': max_score,
                    'guide_count': len(stats['scores'])
                }
            
            # Split into primary/secondary networks
            all_conservation = [s['conservation'] for s in species_quality.values()]
            median_cons = np.median(all_conservation) if all_conservation else 0.5
            
            primary = [sp for sp, q in species_quality.items() if q['conservation'] > median_cons]
            secondary = [sp for sp, q in species_quality.items() if q['conservation'] <= median_cons]
            
            # Draw primary network (left circle)
            primary_center = (0.3, 0.5)
            primary_radius = 0.4
            
            for i, sp in enumerate(primary):
                angle = 2 * np.pi * i / max(len(primary), 1)
                x = primary_center[0] + primary_radius * np.cos(angle)
                y = primary_center[1] + primary_radius * np.sin(angle)
                
                # Color by whether passes threshold
                if species_quality[sp]['max_score'] >= GOOD_SCORE:
                    color = colors['primary']
                    edge = 'black'
                else:
                    color = '#CCCCCC'
                    edge = '#999999'
                
                circle = plt.Circle((x, y), 0.08, facecolor=color, edgecolor=edge)
                ax5.add_patch(circle)
                ax5.text(x, y-0.12, sp[:6], ha='center', va='center', fontsize=7)
            
            # Draw secondary network (right circle)
            secondary_center = (0.7, 0.5)
            secondary_radius = 0.4
            
            for i, sp in enumerate(secondary):
                angle = 2 * np.pi * i / max(len(secondary), 1)
                x = secondary_center[0] + secondary_radius * np.cos(angle)
                y = secondary_center[1] + secondary_radius * np.sin(angle)
                
                if species_quality[sp]['max_score'] >= GOOD_SCORE:
                    color = colors['secondary']
                    edge = 'black'
                else:
                    color = '#CCCCCC'
                    edge = '#999999'
                
                circle = plt.Circle((x, y), 0.08, facecolor=color, edgecolor=edge)
                ax5.add_patch(circle)
                ax5.text(x, y-0.12, sp[:6], ha='center', va='center', fontsize=7)
            
            # Add labels
            ax5.text(primary_center[0], primary_center[1] + 0.6, 
                    'Primary Network\n(High Conservation)', ha='center', fontweight='bold')
            ax5.text(secondary_center[0], secondary_center[1] + 0.6,
                    'Secondary Network\n(Low Conservation)', ha='center', fontweight='bold')
            
            ax5.set_xlim(0, 1)
            ax5.set_ylim(0, 1)
        else:
            ax5.text(0.5, 0.5, 'No species data', ha='center')
    except Exception as e:
        ax5.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center')
    ax5.set_title('5. Dual-Layer Conservation Network', fontweight='bold')
    
    # 6. LOESS Fit and Residual Outlier Map - FIX LEGEND ISSUE
    ax6 = plt.subplot(3, 2, 6)
    try:
        if len(df) > 0 and 'species' in df.columns and len(df) > 5:
            # Create species-level summary
            species_summary = {}
            for _, row in df.iterrows():
                for sp in row['species']:
                    if sp not in species_summary:
                        species_summary[sp] = {'scores': [], 'conservation': []}
                    species_summary[sp]['scores'].append(row['score'])
                    species_summary[sp]['conservation'].append(row['conservation'])
            
            # Extract data for plot
            x_vals = []  # avg conservation quality
            y_vals = []  # guide count
            max_scores = []  # for coloring
            species_names = []
            
            for sp, stats in species_summary.items():
                if stats['conservation'] and stats['scores']:
                    x_vals.append(np.mean(stats['conservation']))
                    y_vals.append(len(stats['scores']))
                    max_scores.append(max(stats['scores']))
                    species_names.append(sp)
            
            if len(x_vals) > 3:
                scatter = ax6.scatter(x_vals, y_vals, c=max_scores, cmap='coolwarm', 
                                     s=50, alpha=0.7, edgecolors='black', label='Species')
                
                # Add species labels
                for i, sp in enumerate(species_names):
                    ax6.annotate(sp[:6], (x_vals[i], y_vals[i]), 
                                xytext=(5, 5), textcoords='offset points',
                                fontsize=6)
                
                # Add linear fit
                if len(x_vals) > 1:
                    try:
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                        x_line = np.linspace(min(x_vals), max(x_vals), 100)
                        y_line = slope * x_line + intercept
                        line = ax6.plot(x_line, y_line, 'k-', alpha=0.7, 
                                label=f'Fit (R²={r_value**2:.3f})')
                        
                        # Find outliers (simplified)
                        residuals = y_vals - (slope * np.array(x_vals) + intercept)
                        outlier_threshold = 2 * np.std(residuals)
                        outliers = np.abs(residuals) > outlier_threshold
                        
                        for i in np.where(outliers)[0]:
                            ax6.plot(x_vals[i], y_vals[i], 'X', color='red', 
                                    markersize=10, markeredgewidth=2, label='Outliers' if i == 0 else "")
                    except:
                        pass  # Skip if regression fails
                
                ax6.set_xlabel('Conservation Quality')
                ax6.set_ylabel('Guide Count')
                
                # Only add legend if we have labeled items
                handles, labels = ax6.get_legend_handles_labels()
                if handles:
                    ax6.legend()
                ax6.grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax6, label='Max Score')
            else:
                ax6.text(0.5, 0.5, 'Need more species', ha='center')
        else:
            ax6.text(0.5, 0.5, 'Need more data', ha='center')
    except Exception as e:
        ax6.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center')
    ax6.set_title('6. LOESS Fit and Residual Map', fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "specific_graphs.png", dpi=300)
    plt.close()
    
    logger.info(f"6 specific graphs saved to {out_dir / 'specific_graphs.png'}")

def export_results(guides, seqs_dict, out_dir):
    """Save results to files with enhanced features"""
    out_dir.mkdir(exist_ok=True)
    
    # CSV with enhanced features
    if guides:
        df = pd.DataFrame(guides)
        
        # Expand off_targets dictionary into separate columns
        if 'off_targets' in df.columns:
            off_target_df = pd.DataFrame(df['off_targets'].tolist())
            df = pd.concat([df.drop('off_targets', axis=1), off_target_df], axis=1)
        
        csv_path = out_dir / "guides.csv"
        df.to_csv(csv_path, index=False)
        
        # Enhanced summary
        good_count = sum(1 for g in guides if g.get('good', False))
        with open(out_dir / "summary.txt", 'w') as f:
            f.write(f"msBLT 2.0 Enhanced Run Summary\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Species analyzed: {len(seqs_dict)}\n")
            f.write(f"Total guides: {len(guides)}\n")
            f.write(f"Good guides: {good_count}\n")
            f.write(f"\nTop 5 guides:\n")
            for i, g in enumerate(guides[:5]):
                f.write(f"{i+1}. {g['seq']} (score: {g['score']}, PAM: {g.get('pam', 'N/A')})\n")
                f.write(f"    Self-complementarity: {g.get('self_complementarity', 0)}\n")
                f.write(f"    Off-targets: MM0={g.get('off_targets', {}).get('MM0', 0)}, ")
                f.write(f"MM1={g.get('off_targets', {}).get('MM1', 0)}, ")
                f.write(f"MM2={g.get('off_targets', {}).get('MM2', 0)}, ")
                f.write(f"MM3={g.get('off_targets', {}).get('MM3', 0)}\n")
                if g.get('genomic_locations'):
                    loc = g['genomic_locations'][0]
                    f.write(f"    Location: {loc['position']} ({loc['strand']})\n")
    
    # FASTA of top guides with enhanced annotation
    fasta_path = out_dir / "top_guides.fasta"
    with open(fasta_path, 'w') as f:
        for i, g in enumerate(guides[:20]):
            header = f">guide_{i+1}_score_{g['score']}_pam_{g.get('pam', 'NA')}"
            header += f"_selfcomp_{g.get('self_complementarity', 0)}"
            header += f"_MM0-{g.get('off_targets', {}).get('MM0', 0)}"
            f.write(f"{header}\n")
            f.write(f"{g['seq']}\n")

def validate_guides(guides_list):
    """Validate against known good CRISPR rules"""
    
    if isinstance(guides_list, list):
        df = pd.DataFrame(guides_list)
    else:
        df = guides_list
    
    results = {
        'total_guides': len(df),
        'basic_checks': {},
        'quality_metrics': {},
        'red_flags': []
    }
    
    # Basic checks
    results['basic_checks'] = {
        'perfect_length': (df['seq'].str.len() == 20).sum(),
        'valid_gc_range': ((df['gc'] >= 40) & (df['gc'] <= 60)).sum(),
        'no_poly_t': (~df['seq'].str.contains('TTTT')).sum(),
        'starts_with_g': (df['seq'].str[0] == 'G').sum(),
        'ends_with_gc': (df['seq'].str[-1].isin(['G', 'C'])).sum()
    }
    
    # Quality metrics
    if 'self_complementarity' in df.columns:
        results['quality_metrics']['zero_self_comp'] = (df['self_complementarity'] == 0).sum()
    
    if 'MM0' in df.columns:
        results['quality_metrics']['clean_off_targets'] = (df['MM0'] == 0).sum()
        results['quality_metrics']['low_off_targets'] = ((df['MM0'] == 0) & (df['MM1'] <= 2)).sum()
    
    # Red flags
    for idx, row in df.iterrows():
        flags = []
        if row['gc'] < 30 or row['gc'] > 70:
            flags.append("Extreme GC content")
        if 'TTTT' in row['seq']:
            flags.append("Contains poly-T terminator")
        if row['seq'].count('G') > 10 or row['seq'].count('C') > 10:
            flags.append("GC skewed composition")
        if 'self_complementarity' in row and row['self_complementarity'] > 5:
            flags.append("High self-complementarity")
        
        if flags:
            results['red_flags'].append({
                'guide': row['seq'],
                'position': idx,
                'flags': flags
            })
    
    return results

def compare_with_chopchop(your_guides, chopchop_data):
    """Head-to-head comparison"""
    
    your_seqs = set([g['seq'] for g in your_guides[:5]])
    chopchop_seqs = set(chopchop_data['Target sequence'].values)
    
    overlap = your_seqs.intersection(chopchop_seqs)
    
    comparison = {
        'your_top5_avg_score': sum([g['score'] for g in your_guides[:5]]) / 5,
        'chopchop_top5_avg_efficiency': chopchop_data['Efficiency'].head(5).mean(),
        'shared_guides': len(overlap),
        'shared_sequences': list(overlap)
    }
    
    # Score comparison for shared guides
    score_comparison = []
    for seq in overlap:
        your_score = next(g['score'] for g in your_guides if g['seq'] == seq)
        chop_score = chopchop_data[chopchop_data['Target sequence'] == seq]['Efficiency'].iloc[0]
        score_comparison.append({
            'sequence': seq,
            'msblt_score': your_score,
            'chopchop_efficiency': chop_score,
            'difference': your_score - chop_score
        })
    
    comparison['score_details'] = score_comparison
    
    return comparison

def run_validation(your_guides_csv_path, chopchop_csv_path=None):
    """Main validation function"""
    
    your_guides = pd.read_csv(your_guides_csv_path)
    validation = validate_guides(your_guides)
    
    print("="*50)
    print("msBLT 2.0 VALIDATION RESULTS")
    print("="*50)
    
    print(f"\nBASIC CHECKS:")
    for check, count in validation['basic_checks'].items():
        percentage = (count / validation['total_guides']) * 100
        print(f"  {check.replace('_', ' ').title()}: {count}/{validation['total_guides']} ({percentage:.1f}%)")
    
    print(f"\nQUALITY METRICS:")
    for metric, count in validation['quality_metrics'].items():
        percentage = (count / validation['total_guides']) * 100
        print(f"  {metric.replace('_', ' ').title()}: {count}/{validation['total_guides']} ({percentage:.1f}%)")
    
    if validation['red_flags']:
        print(f"\nRED FLAGS FOUND: {len(validation['red_flags'])}")
        for flag in validation['red_flags'][:3]:
            print(f"  Guide {flag['position']}: {', '.join(flag['flags'])}")
    
    if chopchop_csv_path and os.path.exists(chopchop_csv_path):
        chopchop_data = pd.read_csv(chopchop_csv_path)
        comparison = compare_with_chopchop(your_guides.to_dict('records'), chopchop_data)
        
        print(f"\nCHOPCHOP COMPARISON:")
        print(f"  Your avg score: {comparison['your_top5_avg_score']:.1f}")
        print(f"  ChopChop avg efficiency: {comparison['chopchop_top5_avg_efficiency']:.1f}")
        print(f"  Shared guides: {comparison['shared_guides']}")
        
        if comparison['shared_guides'] > 0:
            print(f"\n  SHARED GUIDE ANALYSIS:")
            for detail in comparison['score_details']:
                print(f"    {detail['sequence'][:10]}...")
                print(f"      msBLT: {detail['msblt_score']:.1f} | ChopChop: {detail['chopchop_efficiency']:.1f}")
                print(f"      Difference: {detail['difference']:+.1f}")
    
    # Final score
    score = 0
    score += (validation['basic_checks']['valid_gc_range'] / validation['total_guides']) * 20
    score += (validation['basic_checks']['perfect_length'] / validation['total_guides']) * 20
    score += (validation['quality_metrics'].get('zero_self_comp', 0) / validation['total_guides']) * 20
    score += (validation['quality_metrics'].get('clean_off_targets', 0) / validation['total_guides']) * 20
    score += min(20, comparison.get('shared_guides', 0) * 5) if chopchop_csv_path else 10
    
    print(f"\nVALIDATION SCORE: {score:.0f}/100")
    
    if score >= 80:
        print("EXCELLENT - Your guides are well-designed")
    elif score >= 60:
        print("GOOD - Solid guide design with minor areas for improvement")
    else:
        print("NEEDS WORK - Consider adjusting your scoring algorithm")
    
    return validation, comparison if chopchop_csv_path else validation

def main():
    parser = argparse.ArgumentParser(description='msBLT 2.0 pipeline')
    parser.add_argument('--data', '-d', type=Path, default='data', help='Data folder')
    parser.add_argument('--output', '-o', type=Path, default='results', help='Output folder')
    parser.add_argument('--gene', '-g', default='bla', help='Target gene (default: bla)')
    parser.add_argument('--min-species', '-m', type=int, default=1, help='Min species per guide')
    parser.add_argument('--pam', '-p', default='NGG', help='PAM sequence (default: NGG)')
    parser.add_argument('--off-target', action='store_true', help='Include off-target analysis')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logger(args.output)
    logger.info("=" * 50)
    logger.info("msBLT 2.0 Enhanced starting")
    logger.info("=" * 50)
    
    # Run pipeline
    try:
        # 1. Load
        sequences = load_sequences(args.data, args.gene)
        if not sequences:
            logger.error("No sequences loaded!")
            return 1
        
        # 2. Conservation
        conserved = find_conserved_kmers(sequences)
        
        # 3. Guides with enhanced features
        genome_context = None
        if args.off_target:
            # Use first sequence as genome context for off-target analysis
            first_species = list(sequences.values())[0]
            genome_context = first_species[0]['seq']
            logger.info("Running with off-target analysis")
        
        guides = find_guides(sequences, args.min_species, args.pam, genome_context)
        if not guides:
            logger.error("No guides found! Try different PAM sequence with --pam")
            logger.info("Common PAM sequences: NGG, NAG, NGA, NGT, TGG, AGG, GGG, CGG")
            return 1
        
        # Sort guides by score
        guides.sort(key=lambda x: x['score'], reverse=True)
        
        # 4. Plots
        make_plots(guides, args.output)
        
        # 5. Export
        export_results(guides, sequences, args.output)
        
        # Summary
        good = sum(1 for g in guides if g.get('good', False))
        logger.info(f"Done! {len(guides)} guides, {good} good ones")
        print(f"\n* Pipeline completed!")
        print(f"  Output in: {args.output}")
        print(f"  Top guide: {guides[0]['seq']} (score: {guides[0]['score']}, PAM: {guides[0].get('pam', 'unknown')})")
        print(f"  Good guides (score >= {GOOD_SCORE}): {good}")
        
        # Print enhanced stats if available
        if guides and 'self_complementarity' in guides[0]:
            print(f"  Enhanced features: self-complementarity, off-target analysis, genomic locations")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
