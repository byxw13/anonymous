Experimental Framework for a Multi-Strategy Verification Method

This repository contains the code for the experiments described in our paper.

📌 Verification Strategies

The following table maps the identifiers used in the code to their corresponding verification method names.
Identifier	Method Description
Forward	Forward Verification
Reverse	Reverse Verification
SC	Self-Consistency
Dual	Forward Verification + Reverse Verification
SC + Reverse	Self-Consistency + Reverse Verification

💻 Environment Setup

    Python Version: Python 3.10.16

🧩 Required Libraries (and their versions):

Library	Version	Notes
requests	2.32.3	-
openai	1.70.0	-
tiktoken	0.8.0	-
pandas	1.5.3	-
numpy	1.26.4	-
json, re, time, os, concurrent.futures	N/A	Standard Python Libraries

🚀 Quick Start

You can use the following commands to create and configure a Conda virtual environment:
Bash

conda create -n venv python=3.10.16
conda activate venv

pip install requests==2.32.3
pip install openai==1.70.0
pip install tiktoken==0.8.0
pip install pandas==1.5.3
pip install numpy==1.26.4

