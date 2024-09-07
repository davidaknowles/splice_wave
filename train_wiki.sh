for MODEL in Conv Convformer Charformer Mamba Transformer
do
  python train_jax.py -m -g wiki $MODEL
done
