# isCCGs – Pan-ESKAPE CRISPR guides against β-lactamases

## Overview
msBLT is a fully in-silico pipeline that designs CRISPR-Cas9 guides hitting ≥ 85 % on-target efficiency and ≤ 5 % off-target probability across *E. coli, K. pneumoniae, A. baumannii, P. aeruginosa, S. aureus*. The final library contains **188 conserved guides** ready for synthesis.

## Quick start
```bash
git clone https://github.com/Jamesjelo/msBLT-Pipeline.git
cd isCCGs-β-lactamase-guides
docker build -t msblt .
docker run --rm -v $(pwd)/data:/app/data msblt
