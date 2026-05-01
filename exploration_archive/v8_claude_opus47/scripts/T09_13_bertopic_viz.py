"""T09 Step 13: Built-in BERTopic visualizations (topics / hierarchy / barchart).

BERTopic returns Plotly figures; we write them as PNG via plotly.io (kaleido)
if available, else write HTML.
"""
import os
import numpy as np
import pandas as pd
from bertopic import BERTopic

OUTDIR = "exploration/artifacts/T09"
FIGS = "exploration/figures/T09"
os.makedirs(FIGS, exist_ok=True)


def main():
    model = BERTopic.load(f"{OUTDIR}/bertopic_model")
    print(f"Loaded BERTopic model, topics: {len(model.get_topics())}")

    # Try PNG via kaleido
    to_png = True
    try:
        import plotly.io as pio  # noqa
    except ImportError:
        to_png = False

    def save(fig, name):
        try:
            if to_png:
                fig.write_image(f"{FIGS}/{name}.png", width=1200, height=900, scale=2)
                print(f"Wrote PNG: {name}")
            else:
                raise RuntimeError("no kaleido")
        except Exception as e:
            print(f"PNG write failed ({e}); writing HTML: {name}")
            fig.write_html(f"{FIGS}/{name}.html")

    # visualize_topics (inter-topic distance map)
    try:
        fig = model.visualize_topics()
        save(fig, "bertopic_topic_map")
    except Exception as e:
        print(f"visualize_topics failed: {e}")

    # visualize_hierarchy
    try:
        fig = model.visualize_hierarchy()
        save(fig, "bertopic_hierarchy")
    except Exception as e:
        print(f"visualize_hierarchy failed: {e}")

    # visualize_barchart (top terms per topic)
    try:
        n_topics = len([t for t in model.get_topics() if t != -1])
        fig = model.visualize_barchart(top_n_topics=min(n_topics, 16), n_words=10)
        save(fig, "bertopic_barchart")
    except Exception as e:
        print(f"visualize_barchart failed: {e}")

    # Also build matplotlib static barchart for paper-quality
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        topic_info = model.get_topic_info()
        valid_topics = topic_info[topic_info.Topic != -1].head(16)
        names = pd.read_csv(f"{OUTDIR}/archetype_names.csv").set_index("archetype_id")["archetype_name"].to_dict()
        n = len(valid_topics)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.6 * rows))
        for i, (_, row) in enumerate(valid_topics.iterrows()):
            ax = axes.flatten()[i]
            tid = row["Topic"]
            words = model.get_topic(tid)[:10]
            words = words[::-1]  # larger on top
            ax.barh(range(len(words)), [w[1] for w in words], color="#3a7dc6")
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels([w[0] for w in words], fontsize=8)
            pretty = names.get(tid, f"Topic {tid}")
            ax.set_title(f"{tid}: {pretty}\n(n={row['Count']})", fontsize=9)
            ax.set_xticks([])
        # Hide unused
        for j in range(n, rows * cols):
            axes.flatten()[j].set_visible(False)
        fig.suptitle("BERTopic — top 10 c-TF-IDF terms per archetype (mts=30, seed=20260417)", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(f"{FIGS}/bertopic_barchart_static.png", dpi=150, bbox_inches="tight")
        print("Wrote bertopic_barchart_static.png")
    except Exception as e:
        print(f"static barchart failed: {e}")


if __name__ == "__main__":
    main()
