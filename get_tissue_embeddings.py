
model_id = "sentence-transformers/all-MiniLM-L6-v2"
hf_token = "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"


api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

all_tissues = bed_data["tissue"].unique().tolist()

embeds = [ query(g) for g in all_tissues ]

embed_df = pd.DataFrame(embeds)

embed_df.index = all_tissues



embed_df.to_csv(vertebrate_epigenomes / "tissue_embeds.tsv", sep = "\t")

reload_df = pd.read_csv(vertebrate_epigenomes / "tissue_embeds.tsv", sep = "\t", index_col = 0)

for i in range(len(all_tissues)): 
    if np.isnan(embed_df[1].iloc[i]): 
        print(embed_df.index[i])
        break
        #embed_df.iloc[i,:] = query(embed_df.index[i])

embed_df.iloc[i,:] = query("None")
embed_df[0] = embed_df[0].astype(np.float64)

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(embed_df)
X_embedded.shape

import plotnine as p9

df = pd.DataFrame(X_embedded, columns=['tSNE1', 'tSNE2'])
df['label'] = embed_df.index

import re

immune_pattern = re.compile(r"[Ss]pleen|[Tt]hymus|CD|cd|[TtBb] cell|marrow|cytotoxic|macrophages", re.IGNORECASE)
df['label'] = df['label'].astype(str)

df["immune"] = df['label'].apply(lambda x: bool(immune_pattern.search(x)))

plot = (p9.ggplot(df, p9.aes(x='tSNE1', y='tSNE2', color = "immune")) +
        p9.geom_point(size=2, alpha=0.6) + 
        #p9.geom_text() +
        p9.theme_minimal())
plot
