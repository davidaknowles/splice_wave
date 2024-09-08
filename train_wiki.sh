#for MODEL in Conv Convformer Charformer Mamba Transformer
for MODEL in Transformer Charformer 
do
  python train_jax.py -m -g wiki $MODEL
done
