CUDA="cpu"
TORCH_VERSION="1.9.0"
pip install -r requirements.txt
pip install torch==$TORCH_VERSION -f https://download.pytorch.org/whl/$CUDA/torch_stable.html &&
pip install torch-scatter -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA.html &&
pip install torch-sparse -f https://data.pyg.org/whl/torch-$TORCH_VERSION+$CUDA.html &&
pip install torch-geometric &&
pip install torch-geometric-temporal
