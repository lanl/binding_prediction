# Binding Enthalpy Machine Learning Model
This is a trained Gradient-Boosting Trees model and accompanying python script from our paper "Design of Amine-Functionalized Materials for Direct Air Capture Using Integrated High-Throughput Calculations and Machine Learning" (https://arxiv.org/abs/2410.13982)


## Running the model

### 1. Clone the repository 

```bash
git clone https://github.com/lanl/binding_prediction.git
cd binding_prediction/binding_prediction
```

### 2. Set up the conda environment

```bash
conda create -n bind_pred -c conda-forge rdkit mordred numpy=1.23 scikit-learn=1.3.0
conda activate bind_pred
pip install rdchiral
```

### 3. Run the model

```bash
python run_model.py smiles.txt
```

This will print a list of binding enthalpies for each SMILES in "smiles.txt" with the following format, one per line:

\<parent SMILES\>,\<child SMILES\>,\<binding enthalpy\>

If a SMILES has multiple binding sites (as described in the paper) then there will be a line for each unique child.


"smiles.txt" is provided as a small test case. Replace with your own text file (one SMILES per line) to run the model on other molecules.

## License

Approved for open release (O4953).

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration. All rights in the program are
reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear
Security Administration. The Government is granted for itself and others acting on its behalf a
nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare
derivative works, distribute copies to the public, perform publicly and display publicly, and to permit
others to do so.

Copyright 2025
 
Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
