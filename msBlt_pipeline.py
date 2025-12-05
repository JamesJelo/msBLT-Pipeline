# ==================================================================================
# msBLT: Multi-Species Beta-Lactamase Targeting Pipeline
# Howard High-School – Garcia, James Jelo
# ==================================================================================

# ==================================================================================
# CRITICAL PARAMETERS 
# ==================================================================================

DATA_FOLDER = "data"
OUTPUT_DIR  = "result"
GENE_FILTER = "bla"

GUIDE_LENGTH = 20
PAM_SEQUENCE = "NGG"
MIN_SPECIES  = 2
GOOD_SCORE   = 85
OFFTARGET_RISK_CUTOFF = 5.0

MIN_SEQUENCE_LENGTH = 23
VALID_BASES = {"A", "T", "G", "C", "N"}

JSD_CUTOFF_STRICT          = 0.15
MIN_CONSERVED_LENGTH_STRICT = 18
MAX_GAP_COLUMNS_STRICT      = 3
MIN_CONSERVED_FRACTION_STRICT = 0.65

RELAXATION_LEVELS = [
    {"name": "Moderate",        "jsd": 0.12, "length": 16, "gaps": 4, "fraction": 0.55},
    {"name": "Permissive",      "jsd": 0.10, "length": 14, "gaps": 6, "fraction": 0.45},
    {"name": "Ultra-Permissive", "jsd": 0.08, "length": 12, "gaps": 8, "fraction": 0.35},
]

MAX_RELAXATION_LEVEL = len(RELAXATION_LEVELS) - 1
OPTIMAL_GC_RANGE     = (25, 75)
MAX_OFFTARGET_MISMATCHES = 4
SEED_LENGTH = 12
PLOT_DPI    = 600

# ==================================================================================
# IMPORTS
# ==================================================================================

import os, sys, argparse, logging, json, warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any, NamedTuple, Union
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy.stats import entropy, mannwhitneyu, pearsonr, t, sem
from scipy.signal import savgol_filter
import statsmodels.api as sm
from Bio import SeqIO, AlignIO, Align
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from Bio.Data import CodonTable

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)8s | %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)  # Suppress runtime warnings for cleaner output

sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 11, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10,
                     'figure.titlesize': 18, 'savefig.dpi': PLOT_DPI, 'savefig.bbox': 'tight'})

# ==================================================================================
# DATA STRUCTURES
# ==================================================================================

class ConservationWindow(NamedTuple):
    start: int; end: int; jsd_scores: List[float]; gap_columns: int
    species_count: int; mean_jsd: float; std_jsd: float

class GuideCandidate(NamedTuple):
    sequence: str; pam: str; position: int; window: ConservationWindow
    species_list: List[str]; conservation_score: float; emergency_mode: bool = False
    gc_percent: Optional[float] = None; disruption_probability: Optional[float] = None

class DataQualityMetrics(NamedTuple):
    total_sequences: int; total_base_pairs: int; avg_length: float; std_length: float
    gc_distribution: Dict[str, float]; gene_coverage: float; low_quality_count: int
    species_loaded: int; files_processed: int

# ==================================================================================
# SEQUENCE VALIDATION & LOADING
# ==================================================================================

def validate_dna_sequence(seq: str) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
    if not seq or not isinstance(seq, str):
        return False, "Empty or invalid sequence", None
    cleaned = ''.join(seq.upper().split())
    invalid_chars = set(cleaned) - VALID_BASES
    if invalid_chars:
        return False, f"Invalid characters: {sorted(invalid_chars)}", None
    if len(cleaned) < MIN_SEQUENCE_LENGTH:
        return False, f"Too short: {len(cleaned)} < {MIN_SEQUENCE_LENGTH}", None
    gc_count = cleaned.count('G') + cleaned.count('C')
    gc_percent = (gc_count / len(cleaned)) * 100 if cleaned else 0
    quality_metrics = {
        'length': len(cleaned),
        'gc_percent': gc_percent,
        'n_count': cleaned.count('N'),
        'complexity': len(set(cleaned)) / len(VALID_BASES) if len(VALID_BASES) > 0 else 0
    }
    return True, cleaned, quality_metrics

def load_sequences(data_folder: str, gene_filter: Optional[str] = GENE_FILTER, debug: bool = True):
    data_path = Path(data_folder)
    if not data_path.is_dir():
        raise FileNotFoundError(f"Data folder not found: {data_path.absolute()}")
    fasta_files = sorted(list(data_path.glob("*.fasta")) + list(data_path.glob("*.fa")))
    if not fasta_files:
        logger.error("ERROR: No FASTA files found in '%s'", data_folder)
        return {}, DataQualityMetrics(0, 0, 0, 0, {}, 0, 0, 0, 0)
    logger.info("Found %d FASTA files in '%s'", len(fasta_files), data_folder)
    all_sequences: Dict[str, List[Seq]] = {}
    quality_stats = {'total_sequences': 0, 'total_base_pairs': 0, 'lengths': [], 'gc_percents': [], 'low_quality': 0}
    files_processed = 0
    for fasta_file in fasta_files:
        species_name = fasta_file.stem
        sequences: List[Seq] = []
        try:
            record_count = 0; valid_count = 0; gene_found = False
            if debug: logger.info("  Processing: %s", fasta_file.name)
            for record in SeqIO.parse(fasta_file, "fasta"):
                record_count += 1
                if gene_filter and gene_filter.lower() != "none":
                    description_lower = record.description.lower()
                    filter_lower = gene_filter.lower()
                    if filter_lower not in description_lower:
                        if debug and record_count <= 3:
                            logger.debug("    Skipping '%s...' (no '%s' in description)", record.description[:50], gene_filter)
                        continue
                gene_found = True
                is_valid, result, metrics = validate_dna_sequence(str(record.seq))
                if is_valid:
                    sequences.append(Seq(result)); valid_count += 1
                    quality_stats['total_sequences'] += 1; quality_stats['total_base_pairs'] += metrics['length']
                    quality_stats['lengths'].append(metrics['length']); quality_stats['gc_percents'].append(metrics['gc_percent'])
                    if metrics['complexity'] < 0.3 or metrics['n_count'] > len(result) * 0.05:
                        quality_stats['low_quality'] += 1
                        if debug:
                            logger.warning("    Low quality sequence in %s: %s", species_name, record.description[:50])
                    if debug and valid_count <= 3:
                        logger.info("    Valid: %s (length: %d, GC: %.1f%%)", record.description[:40], metrics['length'], metrics['gc_percent'])
                else:
                    if debug: logger.warning("    Invalid: %s", result)
            if sequences:
                all_sequences[species_name] = sequences; files_processed += 1
                logger.info("  %s: loaded %d β-lactamase sequences (%d records scanned)", species_name, len(sequences), record_count)
            else:
                if not gene_found:
                    logger.warning("  %s: No β-lactamase genes found (filter: '%s')", species_name, gene_filter)
                else:
                    logger.warning("  %s: All sequences failed quality control", species_name)
        except Exception as e:
            logger.error("  Error loading %s: %s", fasta_file, e); continue
    if not all_sequences:
        logger.error("\n" + "=" * 70); logger.error("CRITICAL ERROR: No valid β-lactamase sequences loaded!")
        logger.error("=" * 70); logger.error("Possible causes:"); logger.error("1. Wrong folder path: '%s'", data_folder)
        logger.error("2. No FASTA files (*.fasta or *.fa) found"); logger.error("3. Gene filter '%s' didn't match any sequence descriptions", gene_filter)
        logger.error("4. All sequences failed quality control (too short or invalid characters)"); logger.error("\nSOLUTIONS:")
        logger.error("- Check that your FASTA files are in the '%s' folder", data_folder); logger.error("- Open your FASTA files and verify they contain 'bla' in the description")
        logger.error("- Try running with --gene-filter None to load all sequences"); logger.error("- Check sequence length (must be ≥%d bp)", MIN_SEQUENCE_LENGTH); logger.error("=" * 70)
        return {}, DataQualityMetrics(0, 0, 0, 0, {}, 0, 0, 0, 0)
    lengths = quality_stats['lengths']; gc_percents = quality_stats['gc_percents']
    quality_metrics = DataQualityMetrics(
        total_sequences=quality_stats['total_sequences'], total_base_pairs=quality_stats['total_base_pairs'],
        avg_length=np.mean(lengths) if lengths else 0, std_length=np.std(lengths) if lengths else 0,
        gc_distribution={'mean': np.mean(gc_percents) if gc_percents else 0, 'std': np.std(gc_percents) if gc_percents else 0,
                         'min': min(gc_percents) if gc_percents else 0, 'max': max(gc_percents) if gc_percents else 0},
        gene_coverage=len(all_sequences) / len(fasta_files), low_quality_count=quality_stats['low_quality'],
        species_loaded=len(all_sequences), files_processed=files_processed)
    logger.info("\n" + "=" * 60); logger.info("DATA QUALITY SUMMARY"); logger.info("=" * 60)
    logger.info("FASTA files processed: %d", quality_metrics.files_processed); logger.info("Species with β-lactamase genes: %d", quality_metrics.species_loaded)
    logger.info("Total sequences loaded: %d", quality_metrics.total_sequences); logger.info("Total base pairs: %d", quality_metrics.total_base_pairs)
    logger.info("Average length: %.1f +/- %.1f bp", quality_metrics.avg_length, quality_metrics.std_length)
    logger.info("GC content: %.1f ± %.1f%%", quality_metrics.gc_distribution['mean'], quality_metrics.gc_distribution['std'])
    logger.info("Gene coverage: %d/%d files (%.1f%%)", quality_metrics.species_loaded, len(fasta_files), quality_metrics.gene_coverage * 100)
    if quality_metrics.low_quality_count > 0: logger.info("Low quality sequences filtered: %d", quality_metrics.low_quality_count)
    return all_sequences, quality_metrics

# ==================================================================================
# ALIGNMENT & CONSERVATION ANALYSIS
# ==================================================================================

def run_progressive_alignment(sequences: List[Seq]) -> Optional[MultipleSeqAlignment]:
    if len(sequences) < 2:
        logger.warning("Need at least 2 sequences for alignment, got %d", len(sequences)); return None
    try:
        logger.info("Performing progressive alignment with %d sequences", len(sequences))
        seq_records = [SeqRecord(seq, id=f"seq_{i}", description="") for i, seq in enumerate(sequences)]
        aligner = Align.PairwiseAligner(); aligner.mode = 'global'; aligner.match_score = 2
        aligner.mismatch_score = -1; aligner.open_gap_score = -0.5; aligner.extend_gap_score = -0.5
        msa = MultipleSeqAlignment([seq_records[0]])
        for i in range(1, len(seq_records)):
            alignment = aligner.align(str(seq_records[0].seq), str(seq_records[i].seq))[0]
            aligned_seq = SeqRecord(Seq(str(alignment[1])), id=f"seq_{i}", description="")
            if len(str(alignment[1])) > msa.get_alignment_length():
                for record in msa:
                    while len(record.seq) < len(str(alignment[1])): record.seq += '-'
            msa.append(aligned_seq)
        logger.info("Alignment complete: %d sequences, %d positions", len(msa), msa.get_alignment_length())
        return msa
    except Exception as e:
        logger.error("Alignment failed: %s", e, exc_info=True); return None

def calculate_true_jsd_per_column(alignment: MultipleSeqAlignment) -> List[float]:
    jsd_scores = []; align_len = alignment.get_alignment_length()
    for pos in range(align_len):
        column = alignment[:, pos]; counts = np.array([column.upper().count(base) for base in "ATGC"]); total = counts.sum()
        if total == 0: jsd_scores.append(0.0); continue
        freq = counts / total; nonzero_freq = freq[freq > 0]
        if len(nonzero_freq) == 1: jsd_scores.append(0.0)
        else:
            uniform = np.ones(len(nonzero_freq)) / len(nonzero_freq)
            m = 0.5 * (nonzero_freq + uniform)
            jsd = 0.5 * entropy(nonzero_freq, m) + 0.5 * entropy(uniform, m)
            jsd_scores.append(max(0.0, min(1.0, jsd)))
    logger.info("Calculated %d conservation scores", len(jsd_scores)); return jsd_scores

def find_conserved_windows_core(jsd_scores, alignment, species_count, jsd_cutoff, min_length, max_gaps, min_fraction):
    windows = []; align_len = len(jsd_scores)
    for start in range(0, align_len - min_length + 1):
        window_jsd = jsd_scores[start:start + min_length]
        window_cols = alignment[:, start:start + min_length]
        gap_count = sum(1 for i in range(min_length) if '-' in window_cols[:, i])
        conserved_count = sum(1 for jsd in window_jsd if jsd <= jsd_cutoff)
        conserved_fraction = conserved_count / min_length
        if conserved_fraction >= min_fraction and gap_count <= max_gaps:
            windows.append(ConservationWindow(
                start=start, end=start + min_length, jsd_scores=window_jsd,
                gap_columns=gap_count, species_count=species_count,
                mean_jsd=np.mean(window_jsd), std_jsd=np.std(window_jsd)))
    return windows

def try_conservation_analysis(all_sequences, min_species=MIN_SPECIES, fallback_to_relaxed=True, min_windows_required=1):
    metadata = {"original_parameters_used": False, "relaxation_level_applied": None,
                "emergency_mode_activated": False, "parameters_tried": []}
    all_seqs_flat = []
    for species, seqs in all_sequences.items(): all_seqs_flat.extend(seqs)
    if len(all_seqs_flat) < 2:
        logger.error("Need at least 2 sequences for alignment, got %d", len(all_seqs_flat)); return [], None, metadata
    alignment = run_progressive_alignment(all_seqs_flat)
    if alignment is None: return [], None, metadata
    jsd_scores = calculate_true_jsd_per_column(alignment)
    parameter_levels = [{"name": "Original", "jsd": JSD_CUTOFF_STRICT, "length": MIN_CONSERVED_LENGTH_STRICT,
                         "gaps": MAX_GAP_COLUMNS_STRICT, "fraction": MIN_CONSERVED_FRACTION_STRICT}, *RELAXATION_LEVELS]
    for idx, params in enumerate(parameter_levels):
        windows = find_conserved_windows_core(jsd_scores, alignment, len(all_sequences),
                                              params["jsd"], params["length"], params["gaps"], params["fraction"])
        metadata["parameters_tried"].append({"level": params["name"], "windows_found": len(windows)})
        logger.info("  %s parameters: %d windows found", params["name"], len(windows))
        if len(windows) >= min_windows_required:
            if idx == 0: metadata["original_parameters_used"] = True; logger.info("SUCCESS: Using original strict parameters")
            else: metadata["relaxation_level_applied"] = params["name"]; logger.warning("SUCCESS: Using relaxed parameters (%s)", params["name"])
            return windows, alignment, metadata
    if fallback_to_relaxed:
        logger.critical("EMERGENCY MODE ACTIVATED: Using full alignment as single window")
        emergency_window = ConservationWindow(
            start=0, end=len(jsd_scores), jsd_scores=jsd_scores,
            gap_columns=0, species_count=len(all_sequences),
            mean_jsd=np.mean(jsd_scores), std_jsd=np.std(jsd_scores))
        metadata["emergency_mode_activated"] = True; metadata["parameters_tried"].append({"level": "EMERGENCY MODE", "windows_found": 1})
        return [emergency_window], alignment, metadata
    return [], alignment, metadata

# ==================================================================================
# GUIDE RNA EXTRACTION
# ==================================================================================

def calculate_jsd_for_position(alignment: MultipleSeqAlignment, start_pos: int, length: int) -> float:
    jsd_scores = []; align_len = alignment.get_alignment_length()
    for offset in range(length):
        if start_pos + offset >= align_len: break
        col = alignment[:, start_pos + offset]; counts = np.array([col.upper().count(base) for base in "ATGC"]); total = counts.sum()
        if total == 0: jsd_scores.append(0.0); continue
        freq = counts / total; nonzero_freq = freq[freq > 0]
        if len(nonzero_freq) == 1: jsd_scores.append(0.0)
        else:
            uniform = np.ones(len(nonzero_freq)) / len(nonzero_freq)
            m = 0.5 * (nonzero_freq + uniform)
            jsd = 0.5 * entropy(nonzero_freq, m) + 0.5 * entropy(uniform, m)
            jsd_scores.append(max(0.0, min(1.0, jsd)))
    return np.mean(jsd_scores) if jsd_scores else 0.0

def predict_frameshift_disruption(guide_sequence: str) -> float:
    disruption_scores = []
    for cut_offset in [3, 4, 5, 6]:
        if cut_offset >= len(guide_sequence): continue
        frame = (len(guide_sequence) - cut_offset) % 3
        disruption_scores.append(0.6 if frame == 0 else 0.95)
    return max(disruption_scores) if disruption_scores else 0.5

def extract_guides_from_conserved(conserved_windows, alignment, all_sequences, pam_seq=PAM_SEQUENCE, emergency_mode=False):
    guides = []
    if not conserved_windows:
        logger.warning("No conserved windows provided for guide extraction"); return guides
    logger.info("\nExtracting guides from %d conserved windows", len(conserved_windows))
    species_index_map = {}; idx_counter = 0
    for species, seqs in all_sequences.items():
        species_index_map[species] = list(range(idx_counter, idx_counter + len(seqs))); idx_counter += len(seqs)
    total_pam_sites = 0; valid_guides = 0
    for window_idx, window in enumerate(conserved_windows):
        search_end = window.end - GUIDE_LENGTH - 3
        if emergency_mode: search_end = alignment.get_alignment_length() - GUIDE_LENGTH - 3
        for pos in range(window.start, max(window.start, search_end)):
            pam_slice = alignment[:, pos + GUIDE_LENGTH:pos + GUIDE_LENGTH + 3]; pam_site = str(pam_slice).upper()
            if len(pam_site) != 3 or pam_site[1:3] != 'GG': continue
            total_pam_sites += 1; guide_seq = ""; is_valid = True
            for offset in range(GUIDE_LENGTH):
                if pos + offset >= alignment.get_alignment_length(): is_valid = False; break
                col = alignment[:, pos + offset]; bases = [b for b in col.upper() if b in 'ATGC']
                if not bases: is_valid = False; break
                guide_seq += max(set(bases), key=bases.count)
            if not is_valid or len(guide_seq) != GUIDE_LENGTH: continue
            if not set(guide_seq).issubset({'A', 'T', 'G', 'C'}): continue
            valid_guides += 1
            covered_species = []
            for species, indices in species_index_map.items():
                if any(idx < len(alignment) and str(alignment[idx].seq)[pos:pos+GUIDE_LENGTH].replace('-', '') == guide_seq for idx in indices):
                    covered_species.append(species)
            if len(covered_species) < MIN_SPECIES: continue
            jsd_score = calculate_jsd_for_position(alignment, pos, GUIDE_LENGTH)
            gc_percent = calculate_gc_percent(guide_seq); disruption_prob = predict_frameshift_disruption(guide_seq)
            guides.append(GuideCandidate(
                sequence=guide_seq, pam=pam_site, position=pos, window=window,
                species_list=sorted(covered_species), conservation_score=jsd_score,
                emergency_mode=emergency_mode, gc_percent=gc_percent, disruption_probability=disruption_prob))
        if (window_idx + 1) % max(1, len(conserved_windows) // 10) == 0:
            logger.info("  Processed %d/%d windows...", window_idx + 1, len(conserved_windows))
    logger.info("Found %d PAM sites -> %d valid guides -> %d cross-species guides", total_pam_sites, valid_guides, len(guides))
    all_covered_species = set()
    for g in guides: all_covered_species.update(g.species_list)
    if len(all_covered_species) < MIN_SPECIES:
        logger.warning("Insufficient species coverage (%d/%d), activating direct scanning", len(all_covered_species), MIN_SPECIES)
        return extract_guides_directly(all_sequences, MIN_SPECIES)
    return guides

def extract_guides_directly(all_sequences: Dict[str, List[Seq]], min_species: int) -> List[GuideCandidate]:
    logger.critical("EMERGENCY DIRECT SCANNING ACTIVATED – Bypassing alignment")
    guides = []; species_guide_db = {}
    for species, seqs in all_sequences.items():
        species_guide_db[species] = {}
        for seq in seqs:
            seq_str = str(seq)
            for pos in range(len(seq_str) - GUIDE_LENGTH - 2):
                if pos + GUIDE_LENGTH + 3 <= len(seq_str):
                    pam = seq_str[pos + GUIDE_LENGTH + 1:pos + GUIDE_LENGTH + 3]
                    if pam == 'GG':
                        guide = seq_str[pos:pos + GUIDE_LENGTH]
                        if len(guide) == GUIDE_LENGTH and set(guide).issubset({'A', 'T', 'G', 'C'}):
                            species_guide_db[species][guide] = species_guide_db[species].get(guide, 0) + 1
    guide_coverage = {}
    for species, guide_dict in species_guide_db.items():
        for guide, count in guide_dict.items():
            if guide not in guide_coverage: guide_coverage[guide] = {"species": [], "total_count": 0}
            guide_coverage[guide]["species"].append(species); guide_coverage[guide]["total_count"] += count
    for guide, data in guide_coverage.items():
        if len(data["species"]) >= min_species:
            gc_percent = calculate_gc_percent(guide); disruption_prob = predict_frameshift_disruption(guide)
            guides.append(GuideCandidate(
                sequence=guide, pam="NGG", position=-1,
                window=ConservationWindow(start=-1, end=-1, jsd_scores=[0.0], gap_columns=0,
                                        species_count=len(data["species"]), mean_jsd=0.0, std_jsd=0.0),
                species_list=sorted(data["species"]), conservation_score=0.0,
                emergency_mode=True, gc_percent=gc_percent, disruption_probability=disruption_prob))
    logger.warning("Emergency scanning found %d cross-species guides", len(guides))
    if len(guides) < min_species:
        logger.error("CRITICAL: Only %d guides across %d species", len(guides), len(all_sequences))
        logger.error("Recommendation: Use less stringent PAM or more related species")
    return guides

# ==================================================================================
# GUIDE SCORING & ANALYSIS
# ==================================================================================

def calculate_gc_percent(sequence: str) -> float:
    if not sequence: return 0.0
    gc_count = sequence.count('G') + sequence.count('C')
    return (gc_count / len(sequence)) * 100

def score_guide_rule_based(guide_sequence: str) -> float:
    if len(guide_sequence) != GUIDE_LENGTH: return 0.0
    sequence = guide_sequence.upper(); score = 100.0
    gc_percent = calculate_gc_percent(sequence)
    if gc_percent < 25: score -= (25 - gc_percent) * 0.5
    elif gc_percent > 75: score -= (gc_percent - 75) * 0.5
    for i in range(len(sequence) - 4):
        if len(set(sequence[i:i+5])) == 1: score -= 20; break
    if sequence[19] == 'G': score -= 10
    if sequence[0] == 'G': score -= 5
    if 'TTTT' in sequence: score -= 15
    seed = sequence[-SEED_LENGTH:]; seed_gc = calculate_gc_percent(seed)
    if seed_gc < 30: score -= 5
    return round(max(0.0, min(100.0, score)), 2)

def estimate_offtarget_risk(guide_sequence: str, all_sequences: Dict[str, List[Seq]]) -> float:
    seed = guide_sequence[-SEED_LENGTH:]; matches = 0; total = 0; seed_len = len(seed)
    for species, seqs in all_sequences.items():
        for seq in seqs:
            seq_str = str(seq); rc_seq = str(seq.reverse_complement())
            for target_seq in [seq_str, rc_seq]:
                if seed not in target_seq: continue
                for pos in range(len(target_seq) - seed_len + 1):
                    if target_seq[pos:pos + seed_len] == seed:
                        if pos + GUIDE_LENGTH <= len(target_seq):
                            full_guide = target_seq[pos:pos + GUIDE_LENGTH]
                            if full_guide != guide_sequence:
                                mismatches = sum(c1 != c2 for c1, c2 in zip(guide_sequence, full_guide))
                                if mismatches <= MAX_OFFTARGET_MISMATCHES: matches += 1
                        total += 1
    risk = (matches / max(total, 1)) * 100
    return round(min(100.0, risk), 3)

def has_termination_signal(sequence: str) -> bool:
    return 'TTTT' in sequence.upper()

# ==================================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# ==================================================================================

def perform_statistical_tests(df_all: pd.DataFrame) -> Dict[str, Any]:
    """Comprehensive statistical testing suite for guide RNA analysis"""
    if df_all.empty:
        logger.warning("No data for statistical analysis"); return {}
    results = {}
    try:
        # 1. Correlation Analysis: Conservation vs. Guide Score
        if 'Conservation_Score' in df_all.columns and 'Score' in df_all.columns:
            conservation_clean = df_all['Conservation_Score'].dropna()
            score_clean = df_all['Score'].loc[conservation_clean.index]
            if len(conservation_clean) > 2:
                corr_coef, p_value = pearsonr(conservation_clean, score_clean)
                results['conservation_score_correlation'] = {
                    'correlation_coefficient': float(corr_coef),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'sample_size': len(conservation_clean)
                }
        # 2. Mann-Whitney U Test: High vs. Low Scoring Guides
        if 'Score' in df_all.columns:
            high_scorers = df_all[df_all['Score'] >= GOOD_SCORE]['Conservation_Score'].dropna()
            low_scorers = df_all[df_all['Score'] < GOOD_SCORE]['Conservation_Score'].dropna()
            if len(high_scorers) > 0 and len(low_scorers) > 0:
                statistic, p_value = mannwhitneyu(high_scorers, low_scorers, alternative='two-sided')
                results['high_vs_low_scorers_test'] = {
                    'mannwhitneyu_statistic': float(statistic),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'high_scorers_n': len(high_scorers),
                    'low_scorers_n': len(low_scorers),
                    'high_scorers_mean_conservation': float(high_scorers.mean()),
                    'low_scorers_mean_conservation': float(low_scorers.mean())
                }
        # 3. Species Coverage Analysis
        if 'Species_Count' in df_all.columns:
            species_counts = df_all['Species_Count']
            results['species_coverage_distribution'] = {
                'mean': float(species_counts.mean()),
                'std': float(species_counts.std()),
                'median': float(species_counts.median()),
                'max': int(species_counts.max()),
                'min': int(species_counts.min()),
                'q25': float(species_counts.quantile(0.25)),
                'q75': float(species_counts.quantile(0.75))
            }
            broad_spectrum = (species_counts >= MIN_SPECIES).sum()
            total_guides = len(df_all)
            if total_guides > 0:
                results['broad_spectrum_proportion'] = {
                    'proportion': float(broad_spectrum / total_guides),
                    'count': int(broad_spectrum),
                    'total': int(total_guides),
                    'expected': 0.5,
                    'significant': broad_spectrum / total_guides > 0.5
                }
        # 4. GC Content Analysis
        if 'GC_Percent' in df_all.columns:
            gc_content = df_all['GC_Percent']
            optimal_gc = ((gc_content >= OPTIMAL_GC_RANGE[0]) & (gc_content <= OPTIMAL_GC_RANGE[1])).sum()
            results['gc_content_analysis'] = {
                'mean_gc': float(gc_content.mean()),
                'std_gc': float(gc_content.std()),
                'optimal_gc_count': int(optimal_gc),
                'optimal_gc_proportion': float(optimal_gc / len(gc_content)),
                'range': OPTIMAL_GC_RANGE
            }
        # 5. Off-target Risk Analysis
        if 'OffTarget_Risk' in df_all.columns:
            off_target_risks = df_all['OffTarget_Risk']
            high_risk_count = (off_target_risks > OFFTARGET_RISK_CUTOFF).sum()
            results['offtarget_risk_analysis'] = {
                'mean_risk': float(off_target_risks.mean()),
                'median_risk': float(off_target_risks.median()),
                'high_risk_count': int(high_risk_count),
                'high_risk_proportion': float(high_risk_count / len(off_target_risks)),
                'risk_threshold': OFFTARGET_RISK_CUTOFF
            }
        # 6. Guide Score Distribution Analysis
        if 'Score' in df_all.columns:
            scores = df_all['Score']
            if len(scores) > 1:
                mean_score = scores.mean(); sem_score = sem(scores)
                confidence_interval = t.interval(0.95, len(scores)-1, loc=mean_score, scale=sem_score)
                results['guide_score_confidence_interval'] = {
                    'mean': float(mean_score),
                    'sem': float(sem_score),
                    'ci_lower': float(confidence_interval[0]),
                    'ci_upper': float(confidence_interval[1]),
                    'n': int(len(scores))
                }
            above_threshold = (scores >= GOOD_SCORE).sum()
            results['threshold_achievement_rate'] = {
                'count': int(above_threshold),
                'rate': float(above_threshold / len(scores)),
                'threshold': GOOD_SCORE,
                'total_guides': int(len(scores))
            }
        # 7. Network Density Analysis
        if 'Species' in df_all.columns:
            all_species = set()
            for species_list in df_all['Species']:
                all_species.update(species_list)
            n_species = len(all_species)
            if n_species > 1:
                potential_pairs = n_species * (n_species - 1) / 2
                actual_connections = 0
                species_guide_map = {}
                for _, row in df_all.iterrows():
                    for species in row['Species']:
                        if species not in species_guide_map:
                            species_guide_map[species] = set()
                        species_guide_map[species].add(row['Guide_Sequence'])
                species_list = list(species_guide_map.keys())
                for i, s1 in enumerate(species_list):
                    for s2 in species_list[i+1:]:
                        shared = len(species_guide_map[s1].intersection(species_guide_map[s2]))
                        if shared > 0: actual_connections += 1
                results['network_density'] = {
                    'species_count': n_species,
                    'potential_connections': int(potential_pairs),
                    'actual_connections': int(actual_connections),
                    'density': float(actual_connections / max(potential_pairs, 1)),
                    'interpretation': 'high' if actual_connections / max(potential_pairs, 1) > 0.3 else 'low'
                }
        logger.info("Statistical analysis complete: %d tests performed", len(results))
    except Exception as e:
        logger.warning("Statistical analysis failed: %s", e); results['error'] = str(e)
    return results

# ==================================================================================
# NEW VISUALIZATION FUNCTIONS – 6 PANELS (Network Graph Restored)
# ==================================================================================

# 1.  Bullet-proof heatmap (no clustering if matrix invalid)
def create_hierarchical_clustering_heatmap(ax, df_all, species_list):
    """Graph 1: Multi-Species efficiency heatmap (no clustering if matrix invalid)"""
    if df_all.empty or len(species_list) < 2:
        ax.text(.5, .5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('1. Efficiency Matrix', fontweight='bold'); return
    guide_ids = df_all['Guide_Sequence'].unique()[:50]
    efficiency_matrix = np.full((len(species_list), len(guide_ids)), np.nan)
    for i, sp in enumerate(species_list):
        for j, g in enumerate(guide_ids):
            row = df_all[df_all['Guide_Sequence'] == g]
            if not row.empty and sp in row['Species'].iloc[0]:
                efficiency_matrix[i, j] = row['Score'].iloc[0]
    im = ax.imshow(efficiency_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
    ax.contour(efficiency_matrix >= GOOD_SCORE, levels=[0.5], colors='k', linewidths=.5)
    ax.set_xlabel('Guide RNAs (Top 50)', fontweight='bold'); ax.set_ylabel('Species', fontweight='bold')
    ax.set_title('1. Efficiency Matrix  (≥85% threshold)', fontweight='bold')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046); cbar.set_label('Efficiency Score (%)', rotation=270, labelpad=15, fontweight='bold')

# 2.  Conservation gradient curve
def create_conservation_gradient_curve(ax, jsd_scores, conserved_windows):
    if not jsd_scores: return
    pos = np.arange(len(jsd_scores)); cons = 1 - np.array(jsd_scores)
    if len(cons) > 10: smooth = savgol_filter(cons, min(51, len(cons)-1) | 1, 3)
    else: smooth = cons
    ax.plot(pos, smooth, color='#2ca02c', lw=2); ax.fill_between(pos, smooth, alpha=.3, color='#2ca02c')
    if len(smooth) > 10:
        sd = np.gradient(np.gradient(smooth)); infl = np.where(np.diff(np.sign(sd)))[0]
        ax.scatter(infl, smooth[infl], color='r', s=50, zorder=5)
    for w in conserved_windows[:5]: ax.axvspan(w.start, w.end, alpha=.2, color='orange')
    ax.axhline(1 - JSD_CUTOFF_STRICT, ls='--', lw=2, color='b', label=f'Strict Threshold ({JSD_CUTOFF_STRICT})')
    ax.set_xlabel('Alignment Position', fontweight='bold'); ax.set_ylabel('Conservation (1-JSD)', fontweight='bold')
    ax.set_title('2. Conservation Gradient Curve\nSecond-Derivative Inflections', fontweight='bold'); ax.legend()

# 3.  Violin-box hybrid
def create_violin_box_hybrid(ax, df_all, species_list):
    if df_all.empty: return
    data, names = [], []
    for sp in species_list:
        sc = df_all[df_all['Species'].apply(lambda x: sp in x)]['Score'].values
        if len(sc): data.append(sc); names.append(sp)
    if not data: return
    parts = ax.violinplot(data, positions=range(len(names)), showmeans=True, showmedians=False, showextrema=False)
    for pc in parts['bodies']: pc.set_facecolor('#1f77b4'); pc.set_alpha(.6); pc.set_edgecolor('k')
    bp = ax.boxplot(data, positions=range(len(names)), widths=.3, patch_artist=True, showfliers=False)
    for patch in bp['boxes']: patch.set_facecolor('w'); patch.set_alpha(.8); patch.set_edgecolor('r'); patch.set_lw(2)
    ax.axhline(GOOD_SCORE, ls='--', lw=2.5, color='r', label=f'Threshold ({GOOD_SCORE}%)')
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Efficiency Score (%)', fontweight='bold')
    ax.set_title('3. Efficiency Distribution per Species\nViolin-Box Hybrid', fontweight='bold'); ax.legend(loc='lower right')

# 4.  Volcano plot
def create_volcano_plot(ax, df_all):
    if df_all.empty: return
    gpi = df_all['Species_Count'] * df_all['Score']; colors = df_all['Conservation_Score']
    sc = ax.scatter(gpi, df_all['Score'], c=colors, s=100, cmap='viridis', alpha=.7, edgecolors='k', lw=.5)
    ax.axhline(GOOD_SCORE, ls='--', lw=2.5, color='r', label=f'Threshold ({GOOD_SCORE}%)')
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046); cbar.set_label('Conservation Score (JSD)', rotation=270, labelpad=15)
    for _, g in df_all.nlargest(3, 'Score').iterrows():
        ax.annotate(f"{g['Guide_Sequence'][:8]}...", (g['Species_Count'] * g['Score'], g['Score']),
                    xytext=(10, 10), textcoords='offset points', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=.7), arrowprops=dict(arrowstyle='->'))
    ax.set_xlabel('Guide Performance Index (Species × Efficiency)', fontweight='bold'); ax.set_ylabel('Efficiency Score (%)', fontweight='bold')
    ax.set_title('4. Multi-Species Volcano Plot\nGPI vs. Efficiency', fontweight='bold'); ax.legend(loc='lower right')

# 5.  Conservation-quality scatter 
def create_conservation_quality_scatter(ax, df_all, species_list):
    """Fixed version with proper NaN/Inf handling and variance checks"""
    if df_all.empty: return
    specs = []
    for sp in species_list:
        g = df_all[df_all['Species'].apply(lambda x: sp in x)]
        if not g.empty:
            cons_mean = g['Conservation_Score'].mean()
            # Only add if we have valid data
            if not pd.isna(cons_mean) and len(g) > 0:
                specs.append({'species': sp, 'cons': cons_mean, 'yield': len(g)})
    if not specs: return
    df = pd.DataFrame(specs)
    
    # Remove any rows with NaN values
    df = df.dropna()
    
    if df.empty: return
    
    ax.scatter(df['cons'], df['yield'], s=100, alpha=.7, c='b', edgecolors='k')
    
    # Only do LOESS if we have enough points and variance (prevents division by zero)
    if len(df) > 5 and df['cons'].var() > 0 and df['yield'].var() > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lowess = sm.nonparametric.lowess(df['yield'], df['cons'], frac=.6, it=3)
                ax.plot(lowess[:, 0], lowess[:, 1], color='r', lw=2, label='LOESS')
        except:
            pass
    
    # Only do linear fit if we have enough points and variance
    if len(df) > 2 and df['cons'].var() > 0 and df['yield'].var() > 0:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                z = np.polyfit(df['cons'], df['yield'], 1)
                p = np.poly1d(z)
                r_squared = np.corrcoef(df["cons"], df["yield"])[0,1]**2
                ax.plot(df['cons'], p(df['cons']), 'r--', alpha=.5, 
                        label=f'Linear Fit (R²={r_squared:.3f})')
        except:
            pass
    
    ax.set_xlabel('Conservation Quality (1-JSD)', fontweight='bold')
    ax.set_ylabel('Guide Yield', fontweight='bold')
    ax.set_title('5. Conservation vs. Guide Yield\nLOESS + Linear Fit', fontweight='bold')
    ax.legend(loc='upper right')

# 6.  Species network graph 
def create_species_network_graph(ax, df_all, species_list):
    """Node size ∝ #guides, edge width ∝ shared guides."""
    if df_all.empty or len(species_list) < 2:
        ax.text(.5, .5, 'Insufficient data\nfor network', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('6. Species Network Graph\n(Guides as edges)', fontweight='bold'); return
    G = nx.Graph()
    for sp in species_list:
        guides = df_all[df_all['Species'].apply(lambda x: sp in x)]
        G.add_node(sp, guides=len(guides))
    for _, row in df_all.iterrows():
        spp = row['Species']
        for i, s1 in enumerate(spp):
            for s2 in spp[i+1:]:
                if G.has_edge(s1, s2): G[s1][s2]['weight'] += 1
                else: G.add_edge(s1, s2, weight=1)
    pos = nx.spring_layout(G, k=3/np.sqrt(len(G.nodes())), iterations=50, seed=42)
    d = dict(G.degree)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=[v*30 for v in d.values()],
                           node_color=list(d.values()), cmap='plasma', alpha=.9)
    nx.draw_networkx_edges(G, pos, ax=ax, width=[G[u][v]['weight']*.5 for u, v in G.edges()],
                           alpha=.6, edge_color='gray')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
    ax.set_title('6. Species Network Graph\nNode size ∝ guides, Edge width ∝ shared guides', fontweight='bold'); ax.axis('off')

# ==================================================================================
# DASHBOARD ASSEMBLY 
# ==================================================================================
def create_isef_dashboard(df_all, species_list, output_dir, jsd_scores, conserved_windows, alignment=None):
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('msBLT: Multi-Species Beta-Lactamase Targeting Analysis\n'
                 'CRISPR-Cas9 Guide Design for Pan-ESKAPE Pathogens', fontsize=22, fontweight='bold', y=.97)
    emergency = len(conserved_windows) == 1 and conserved_windows[0].start == -1
    col = 'red' if emergency else 'green'; txt = "EMERGENCY MODE" if emergency else "Strict Parameters"
    fig.text(.5, .95, f'Status: {txt}', ha='center', fontsize=14, color=col, fontweight='bold',
             bbox=dict(boxstyle='round,pad=.5', facecolor='lightgray', alpha=.8))
    gs = fig.add_gridspec(3, 2, hspace=.35, wspace=.3)
    create_hierarchical_clustering_heatmap(fig.add_subplot(gs[0, 0]), df_all, species_list)
    create_conservation_gradient_curve(fig.add_subplot(gs[0, 1]), jsd_scores, conserved_windows)
    create_violin_box_hybrid(fig.add_subplot(gs[1, 0]), df_all, species_list)
    create_volcano_plot(fig.add_subplot(gs[1, 1]), df_all)
    create_conservation_quality_scatter(fig.add_subplot(gs[2, 0]), df_all, species_list)
    create_species_network_graph(fig.add_subplot(gs[2, 1]), df_all, species_list)
    plt.tight_layout(rect=[0, 0, 1, .96])
    png = out / "msBLT_research_dashboard.png"; pdf = out / "msBLT_research_dashboard.pdf"
    plt.savefig(png, dpi=600, facecolor='white', bbox_inches='tight'); logger.info('Saved: %s', png.absolute())
    plt.savefig(pdf, format='pdf', facecolor='white', bbox_inches='tight'); logger.info('Saved: %s', pdf.absolute())
    plt.close()

# ==================================================================================
# NUMPY->JSON FIX 
# ==================================================================================
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, (np.floating, np.complexfloating)): return float(obj)
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.string_): return str(obj)
        if isinstance(obj, (datetime, date)): return obj.isoformat()
        if isinstance(obj, timedelta): return str(obj)
        return super(NpEncoder, self).default(obj)

# ==================================================================================
# DATA PROCESSING & PIPELINE FUNCTIONS
# ==================================================================================

def process_guides_for_output(guides: List[GuideCandidate], all_sequences: Dict[str, List[Seq]]) -> pd.DataFrame:
    if not guides: logger.warning("No guides to process"); return pd.DataFrame()
    records = []
    for i, guide in enumerate(guides):
        records.append({
            'Rank': i + 1, 'Guide_Sequence': guide.sequence, 'PAM': guide.pam, 'Position': guide.position,
            'Species': guide.species_list, 'Species_Count': len(guide.species_list),
            'Conservation_Score': round(guide.conservation_score, 4),
            'GC_Percent': round(guide.gc_percent, 2) if guide.gc_percent else 0.0,
            'Score': score_guide_rule_based(guide.sequence),
            'OffTarget_Risk': estimate_offtarget_risk(guide.sequence, all_sequences),
            'Disruption_Prob': round(guide.disruption_probability or 0.0, 2),
            'Emergency_Mode': guide.emergency_mode, 'Window_Start': guide.window.start, 'Window_End': guide.window.end
        })
    df = pd.DataFrame(records); df['Has_Termination'] = df['Guide_Sequence'].apply(has_termination_signal); return df

def build_conservation_network(df_all: pd.DataFrame) -> Dict[str, Any]:
    if df_all.empty: return {"primary_network": {}, "secondary_network": {}, "additional_species": []}
    max_species_count = df_all['Species_Count'].max(); primary_guides = df_all[df_all['Species_Count'] == max_species_count]
    primary_species = set(); all_species = set()
    for species_list in primary_guides['Species']: primary_species.update(species_list)
    for species_list in df_all['Species']: all_species.update(species_list)
    additional_species = list(all_species - primary_species)
    return {
        "primary_network": {"species_count": len(primary_species), "guide_count": len(primary_guides), "species_list": sorted(primary_species)},
        "secondary_network": {"species_count": len(all_species), "guide_count": len(df_all), "species_list": sorted(all_species)},
        "additional_species": sorted(additional_species)
    }

def save_detailed_results(df_all, output_dir, species_list, quality_metrics, network_analysis,
                          statistical_results, jsd_scores, conserved_windows, emergency_mode=False):
    out = Path(output_dir); out.mkdir(exist_ok=True, parents=True)
    csv_path = out / "msBLT_guide_results.csv"
    df_export = df_all.copy(); df_export['Species'] = df_export['Species'].apply(lambda x: ';'.join(x))
    df_export.to_csv(csv_path, index=False); logger.info('Saved CSV: %s', csv_path.absolute())
    summary = {
        "timestamp": datetime.now().isoformat(), "total_guides": int(len(df_all)), "species_count": int(len(species_list)),
        "quality_metrics": quality_metrics._asdict(), "emergency_mode": bool(emergency_mode),
        "network_analysis": network_analysis, "statistical_tests": statistical_results,
        "conservation_windows": len(conserved_windows), "avg_jsd": float(np.mean(jsd_scores)) if jsd_scores else 0.0,
        "parameters": {"GUIDE_LENGTH": GUIDE_LENGTH, "PAM_SEQUENCE": PAM_SEQUENCE, "GOOD_SCORE": GOOD_SCORE,
                       "MIN_SPECIES": MIN_SPECIES, "JSD_CUTOFF_STRICT": JSD_CUTOFF_STRICT,
                       "MIN_CONSERVED_LENGTH_STRICT": MIN_CONSERVED_LENGTH_STRICT}
    }
    json_path = out / "msBLT_summary.json"
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, cls=NpEncoder)
    logger.info('Saved JSON: %s', json_path.absolute())

# ==================================================================================
# MAIN PIPELINE
# ==================================================================================
def run_msBLT_pipeline(data_folder=DATA_FOLDER, output_dir=OUTPUT_DIR, gene_filter=GENE_FILTER,
                       min_species=MIN_SPECIES, create_plots=True, save_alignment=False):
    logger.info("=" * 70); logger.info("msBLT Pipeline Starting"); logger.info("=" * 70)
    logger.info("Parameters: data_folder=%s, output_dir=%s, gene_filter=%s, min_species=%d", data_folder, output_dir, gene_filter, min_species)
    logger.info("\n[STEP 1] Loading and validating sequences...")
    all_sequences, quality_metrics = load_sequences(data_folder, gene_filter)
    if not all_sequences: logger.error("Pipeline failed: No valid sequences loaded"); return {"success": False, "error": "No sequences loaded"}
    logger.info("\n[STEP 2] Running conservation analysis...")
    conserved_windows, alignment, conservation_metadata = try_conservation_analysis(all_sequences, min_species, fallback_to_relaxed=True)
    if not conserved_windows: logger.error("Pipeline failed: No conserved windows found"); return {"success": False, "error": "No conserved windows"}
    logger.info("\n[STEP 3] Extracting guide RNAs...")
    emergency_mode = conservation_metadata.get("emergency_mode_activated", False)
    guides = extract_guides_from_conserved(conserved_windows, alignment, all_sequences, emergency_mode=emergency_mode)
    if not guides: logger.error("Pipeline failed: No guides extracted"); return {"success": False, "error": "No guides extracted"}
    logger.info("\n[STEP 4] Processing and analyzing guides...")
    df_all = process_guides_for_output(guides, all_sequences); species_list = sorted(list(all_sequences.keys()))
    network_analysis = build_conservation_network(df_all)
    logger.info("\n[STEP 5] Performing statistical tests...")
    statistical_results = perform_statistical_tests(df_all)
    jsd_scores = calculate_true_jsd_per_column(alignment) if alignment else []
    logger.info("\n[STEP 6] Saving results...")
    save_detailed_results(df_all, output_dir, species_list, quality_metrics, network_analysis,
                          statistical_results, jsd_scores, conserved_windows, emergency_mode)
    if create_plots:
        logger.info("\n[STEP 7] Creating research-focused dashboard...")
        try: create_isef_dashboard(df_all, species_list, output_dir, jsd_scores, conserved_windows, alignment)
        except Exception as e: logger.warning("Dashboard creation failed: %s", e)
    if save_alignment and alignment:
        alignment_path = Path(output_dir) / "msBLT_alignment.fasta"
        try: AlignIO.write(alignment, alignment_path, "fasta"); logger.info("Saved alignment: %s", alignment_path.absolute())
        except Exception as e: logger.warning("Failed to save alignment: %s", e)
    logger.info("\n" + "=" * 70); logger.info("PIPELINE COMPLETE"); logger.info("=" * 70)
    logger.info("Total guides: %d", len(df_all)); logger.info("Species covered: %d", len(species_list))
    if not df_all.empty: logger.info("Top guide score: %.1f", df_all['Score'].max()); logger.info("Guides above threshold: %d/%d", len(df_all[df_all['Score'] >= GOOD_SCORE]), len(df_all))
    failing_species = [s for s in species_list if df_all[df_all['Species'].apply(lambda x: s in x)]['Score'].max() < GOOD_SCORE]
    if failing_species: logger.warning("⚠️ Species below ≥85% threshold: %s", ', '.join(failing_species))
    if emergency_mode: logger.warning("⚠️ EMERGENCY MODE WAS ACTIVATED - Results may be suboptimal")
    else: logger.info("✓ Used strict parameters")
    return {"success": True, "guides": df_all, "species": species_list, "quality_metrics": quality_metrics,
            "conservation_metadata": conservation_metadata, "network_analysis": network_analysis, "statistical_results": statistical_results}

# ==================================================================================
# COMMAND-LINE INTERFACE
# ==================================================================================
def main():
    parser = argparse.ArgumentParser(description="msBLT: Multi-Species Beta-Lactamase Targeting Pipeline")
    parser.add_argument("--data-folder",  default=DATA_FOLDER); parser.add_argument("--output-dir",   default=OUTPUT_DIR)
    parser.add_argument("--gene-filter",  default=GENE_FILTER); parser.add_argument("--min-species",  type=int, default=MIN_SPECIES)
    parser.add_argument("--no-plots",     action='store_true', help="Skip dashboard generation")
    parser.add_argument("--save-alignment", action='store_true', help="Save MSA to FASTA")
    args = parser.parse_args()
    result = run_msBLT_pipeline(data_folder=args.data_folder, output_dir=args.output_dir,
                                gene_filter=args.gene_filter, min_species=args.min_species,
                                create_plots=not args.no_plots, save_alignment=args.save_alignment)
    if not result["success"]: sys.exit(1)

if __name__ == "__main__":
    main()
