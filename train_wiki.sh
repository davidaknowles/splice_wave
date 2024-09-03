for MODEL in Conv Convformer Charformer Mamba Transformer
do
  python train_jax.py -g wiki $MODEL
done
