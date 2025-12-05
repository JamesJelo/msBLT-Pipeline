# isCCGs – Pan-ESKAPE CRISPR guides against β-lactamases

## Overview
msBLT is a fully in-silico pipeline that designs CRISPR-Cas9 guides hitting ≥ 85 % on-target efficiency and ≤ 5 % off-target probability across *E. coli, K. pneumoniae, A. baumannii, P. aeruginosa, S. aureus*. The final library contains **188 conserved guides** ready for synthesis.

## Quick start
```bash
conda env create -f environment.yml
conda activate isCCGs
python -m src.. 01_pangenome_fetch 
