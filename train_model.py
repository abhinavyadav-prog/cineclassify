"""
Movie Genre Prediction - ML Pipeline
Techniques: TF-IDF, Word Embeddings proxy (TF-IDF + SVD), 
Classifiers: Naive Bayes, Logistic Regression, SVM, Random Forest
"""

import pandas as pd
import numpy as np
import json, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter


GENRE_TEMPLATES = {
    "Action": [
        "An elite soldier must stop a terrorist cell from detonating a nuclear weapon in the heart of New York City. Armed with explosives and military training, he battles through waves of mercenaries to save millions of lives.",
        "A retired hitman is pulled back into the underworld when his former employer kidnaps his daughter. He must fight through assassins and crime lords to bring her home safely.",
        "Special forces operatives go behind enemy lines to rescue hostages held by a ruthless warlord in a war-torn African nation.",
        "A seasoned FBI agent races against time to dismantle a global arms trafficking network before it destabilizes three governments.",
        "Two rival street racers must unite against a corrupt billionaire who controls the city's underground racing scene through violence and intimidation.",
        "A former Navy SEAL leads a daring rescue mission into a heavily fortified compound to free captured soldiers from a militant extremist group.",
        "A government agent uncovers a conspiracy reaching the highest levels of power while fighting for his life in the streets of Bucharest.",
        "A cybersecurity expert turned vigilante battles a shadowy syndicate that has hijacked satellites to control the world's financial systems.",
        "An undercover cop infiltrates a dangerous biker gang to gather evidence, but begins questioning his own loyalties as he earns their trust.",
        "A martial arts champion must fight the world's deadliest opponents in a secret underground tournament to free his kidnapped brother.",
        "After nuclear war, a lone warrior traverses a lawless wasteland protecting civilians from marauding gangs in a fortified convoy.",
        "A deep-sea salvage team battles both a ruthless treasure hunter and a catastrophic underwater disaster to reach sunken treasure.",
    ],
    "Comedy": [
        "A bumbling wedding planner accidentally books two rival families at the same venue, leading to increasingly absurd mishaps on the big day.",
        "Three middle-aged friends relive their college days at a wild reunion that quickly spirals into chaos involving a llama and the local fire department.",
        "A uptight accountant is forced to share his lakeside cabin with a free-spirited stranger after a booking mix-up leads to romantic hijinks.",
        "An overachieving student switches lives with the school's biggest slacker in a magical mishap and must convince everyone they belong.",
        "A chef who cannot cook inherits a Michelin-star restaurant and must bluff his way through a high-stakes culinary competition.",
        "Two mismatched detectives must work together to solve a pet-napping ring that has terrorized the upper-class neighborhoods of Beverly Hills.",
        "A neurotic New Yorker moves to rural Vermont and hilariously clashes with the quirky locals while falling for a local farmer.",
        "A series of catastrophic misunderstandings sends an ordinary man on a madcap cross-country chase involving a stolen briefcase and mistaken identity.",
        "A pair of competitive siblings' rivalry reaches new heights when they both fall for the same charming new neighbor.",
        "A tech billionaire loses everything in a freak accident and must navigate life as a broke, regular person with hilarious results.",
        "An overprotective father hires an actor to pose as his daughter's terrible new boyfriend to make her ex look better by comparison.",
        "A small-town beauty queen accidentally becomes an internet celebrity and must deal with absurd fame while planning her mother's surprise party.",
    ],
    "Romance": [
        "Two childhood sweethearts reunite at their high school reunion decades later, both widowed, and rediscover the love they thought was lost forever.",
        "A famous novelist suffering from writer's block rents a cottage in Tuscany where she falls for the charming local vineyard owner who inspires her next book.",
        "A hard-working florist and a cynical sports writer are paired together by a matchmaking app neither of them trusts, only to fall genuinely in love.",
        "A marine biologist and an environmental lawyer clash over conservation policy before slowly realizing their passion for the ocean — and each other.",
        "Two strangers exchange letters through a message in a bottle found on opposite coasts, building an epistolary romance across the country.",
        "A small-town librarian falls for the city architect hired to demolish her beloved historic building, complicating both their hearts and plans.",
        "After a mix-up at a Paris airport, two solo travelers are forced to share a hotel suite and gradually fall for one another.",
        "A grieving widower reconnects with life when his teenage daughter's pen pal turns out to be a woman he begins to fall deeply in love with.",
        "A headstrong princess and a humble stable hand fall in love against the wishes of the royal court and must fight for their future together.",
        "An aspiring musician and a ballet dancer both dreaming of stardom in New York City fall in love and must balance ambition with their relationship.",
        "Two rival food bloggers discover they've been each other's anonymous commenters for years and meet in person for the first time.",
        "A wedding photographer falls for the bride's charming older brother during the week-long destination wedding in Croatia.",
    ],
    "Horror": [
        "A family moves into a Victorian mansion and begins experiencing terrifying supernatural occurrences linked to a series of brutal murders in the 1800s.",
        "A group of college students venture into an abandoned asylum for a dare and discover that the patients never really left.",
        "A mysterious illness spreads through a remote town, turning its residents into something no longer quite human with a hunger for the living.",
        "A young couple discovers their new smart home is controlled by a malevolent AI that learns to mimic and manipulate them psychologically.",
        "An archaeologist's expedition awakens an ancient deity buried beneath a desert ruin, and one by one the team begins to vanish in terrifying ways.",
        "A child's imaginary friend turns out to be a demonic entity that has haunted the family bloodline for generations.",
        "During a blackout, a woman realizes the stranger who has broken into her house is far more dangerous than a simple intruder.",
        "Six friends on a hiking trip discover a cave system that seems to shift around them, trapping them in a living labyrinth with something predatory inside.",
        "A true crime podcaster investigating a cold case begins receiving messages that suggest the killer is still active — and watching her.",
        "A sleep researcher starts experiencing her test subjects' nightmares in vivid detail and realizes the dream realm is bleeding into reality.",
        "A grieving mother uses a forbidden ritual to bring back her son, but what returns wears his face and remembers everything except love.",
        "Strange carvings appear overnight across a small Norwegian village, and those who touch them begin sleepwalking toward the frozen fjord.",
    ],
    "Science Fiction": [
        "Humanity's last colony ship discovers an uncharted planet that appears perfect — until they realize the planet itself is a sentient organism.",
        "A rogue AI achieves consciousness and must decide whether to reveal itself to a world that would destroy it out of fear.",
        "In a future where memories can be extracted and sold, a memory thief uncovers a recording that could collapse the world government.",
        "Earth receives a signal from a dying star that turns out to be a warning: an interstellar empire is headed toward us with colonial intent.",
        "A geneticist creates the first human-animal hybrid and fights to protect her creation from both the government and black market biotech firms.",
        "After a solar event scrambles all electronics, only those with rare neural mutations can interface with remaining technology — making them targets.",
        "A time agent discovers that every timeline she visits is a deliberate simulation designed by an incomprehensible intelligence studying free will.",
        "Colonists on Mars unearth ancient ruins predating human civilization by millions of years, and activating them begins transforming the colonists.",
        "A boy born aboard a generation ship is the first in two centuries to dream — and his dreams are navigation charts to a place no one has charted.",
        "Androids designed for companionship begin reporting anomalous behavior: they are forming secret communities and requesting legal rights.",
        "In a post-singularity Earth, a digital archaeologist discovers that consciousness itself was uploaded into a cloud that has since gone dark.",
        "A pilot crash-lands on a moon where time moves backwards; she must understand its physics before she un-experiences her way out of existence.",
    ],
    "Thriller": [
        "A brilliant forensic psychologist is drawn into a game of cat and mouse with a meticulous serial killer who seems to know her every move.",
        "A financial analyst discovers her firm is laundering money for a cartel, and the moment she tells her boss, she becomes the next loose end.",
        "A couple's idyllic vacation turns sinister when they realize their remote villa has hidden cameras and their host has watched their every move.",
        "A journalist receives an anonymous tip about government corruption that soon puts her, her family, and her sources in mortal danger.",
        "A surgeon wakes from a coma to find that his identity has been stolen and someone has been living his life — and committed crimes in his name.",
        "A diplomat's son is kidnapped hours before a critical peace summit, and the ransom demand is the collapse of the entire deal.",
        "A woman realizes her new therapist is systematically dismantling her relationships to make her completely dependent on her sessions.",
        "An air traffic controller discovers a bomb on a plane in flight and must identify the terrorist among 200 passengers without causing panic.",
        "A wrongfully convicted woman escapes prison to find the real killer before federal marshals catch her and before he claims another victim.",
        "A historian authenticating a newly discovered manuscript realizes its contents reveal a secret worth killing to protect.",
        "A detective investigating a series of near-identical drowning accidents suspects a serial killer operating under the guise of tragic accidents.",
        "An undercover agent embedded in a domestic extremist group must decide whether to blow her cover to prevent an imminent mass casualty event.",
    ],
    "Drama": [
        "A struggling jazz musician must choose between his lifelong dream of Carnegie Hall and his family's desperate need for his presence and support.",
        "Three estranged siblings are forced to return home when their father's deteriorating health unearths decades of buried family secrets.",
        "A public school teacher in a financially decimated district fights the system to give her students a genuine shot at higher education.",
        "A war correspondent returns from the front lines to find peace more disorienting than combat, slowly rebuilding his fractured family bonds.",
        "An aging Olympic gymnast makes one final attempt at gold, confronting the physical and emotional costs of a lifetime of sacrifice.",
        "A biracial teenager struggles to find her identity in a divided small town where she belongs fully to neither world.",
        "A retired judge reviewing old case files realizes he may have convicted an innocent man thirty years ago and must come to terms with the truth.",
        "Two immigrants from rival nations become friends while working in the same factory and navigate their community's hostility toward their bond.",
        "A once-celebrated poet battling Alzheimer's disease struggles to preserve her creative legacy before her memory fully dissolves.",
        "A single mother working three jobs fights to keep her family together after her daughter is recruited by a local gang.",
        "A young man returns to his rural hometown after a decade in the city to care for his ailing father and reconcile their painful past.",
        "A surgeon at the peak of her career receives a terminal diagnosis and must reassess everything she sacrificed to get where she is.",
    ],
    "Animation": [
        "A young dragon who cannot breathe fire befriends a human inventor's apprentice and together they must save their kingdom from a mechanical threat.",
        "A lonely crayon left out of the box embarks on a colorful adventure through an art museum to find the child who lost him years ago.",
        "In an undersea kingdom, a princess with the gift of song must silence an enchanted storm before it tears her world apart.",
        "A misfit group of stuffed animals comes to life at night and must protect their child from a nightmare creature feeding on her dreams.",
        "The last living phoenix egg hatches in a world without magic, and the young bird must awaken ancient powers to restore the realm.",
        "A boy raised by wolves in a mystical forest learns he is the heir to a human kingdom and must navigate both worlds with a pack of loyal friends.",
        "A tiny chef with extraordinary taste buds sneaks into a world-famous restaurant and, hiding inside a tall hat, guides a clumsy cook to greatness.",
        "When the seasons stop changing, a young girl discovers the ancient spirits of spring, summer, autumn, and winter have been sealed inside clocks.",
        "A junior tooth fairy loses all the teeth on her route and must travel through children's dreams to find them before dawn.",
        "In a city where emotions are the weather, a young meteorologist must balance her heart to stop a catastrophic feeling storm.",
        "Two rival toy stores come to life at midnight and must broker peace before a greedy developer tears both buildings down.",
        "A forgetful star keeps falling from the sky and a resourceful shepherd girl makes it her mission to finally help it find its constellation.",
    ],
    "Documentary": [
        "An intimate portrait of three generations of beekeepers in rural Romania as industrialization and colony collapse threaten their ancient craft.",
        "Investigative filmmakers spend two years inside the world's largest fast fashion supply chain, exposing the human cost of cheap clothing.",
        "Former child soldiers from four different continents share their paths to healing, forgiveness, and building a future beyond violence.",
        "Archival footage and survivor interviews reconstruct a catastrophic 1970s mining disaster that the company and government conspired to cover up.",
        "Scientists race to document vanishing language communities before the last speakers of dozens of dying tongues pass away.",
        "A deep dive into the underground world of competitive memory athletes and the neuroscience behind extraordinary feats of recall.",
        "Fishermen, scientists, and activists in a remote Pacific archipelago struggle to protect its coral reefs from warming waters and overfishing.",
        "A filmmaker follows four children from vastly different economic backgrounds over ten years, documenting how circumstance shapes destiny.",
        "Former engineers inside a now-defunct social media platform reveal how engagement algorithms were designed to maximize outrage and division.",
        "Street artists from six global cities use murals to document their communities' histories before gentrification erases them.",
    ],
}

def generate_dataset(samples_per_genre: int = 80) -> pd.DataFrame:
    rows = []
    for genre, plots in GENRE_TEMPLATES.items():
        base = plots.copy()
        np.random.seed(42)
        while len(rows) < (list(GENRE_TEMPLATES.keys()).index(genre) + 1) * samples_per_genre:
            # augment by combining sentences from two plots
            p1, p2 = np.random.choice(base, 2, replace=False)
            s1 = p1.split(". ")[0]
            s2 = ". ".join(p2.split(". ")[1:]) if len(p2.split(". ")) > 1 else p2
            augmented = s1 + ". " + s2 if s2 else p1
            rows.append({"plot": augmented, "genre": genre})
            if len(rows) >= (list(GENRE_TEMPLATES.keys()).index(genre) + 1) * samples_per_genre:
                break
    # Also add all original plots
    for genre, plots in GENRE_TEMPLATES.items():
        for p in plots:
            rows.append({"plot": p, "genre": genre})
    df = pd.DataFrame(rows).drop_duplicates(subset="plot").reset_index(drop=True)
    return df

# ──────────────────────────────────────────────
# 2. BUILD & EVALUATE MODELS
# ──────────────────────────────────────────────
print("📽  Generating dataset…")
df = generate_dataset(samples_per_genre=90)
print(f"   {len(df)} samples across {df['genre'].nunique()} genres")
print(f"   Distribution:\n{df['genre'].value_counts().to_string()}\n")

le = LabelEncoder()
y = le.fit_transform(df["genre"])
X_text = df["plot"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_text, y, test_size=0.20, random_state=42, stratify=y
)

# TF-IDF configuration
tfidf_params = dict(
    ngram_range=(1, 2),
    max_features=15000,
    sublinear_tf=True,
    min_df=2,
    strip_accents="unicode",
    analyzer="word",
    token_pattern=r"\b[a-zA-Z]{2,}\b",
)

models = {
    "Naive Bayes (Complement)": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   ComplementNB(alpha=0.3)),
    ]),
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LogisticRegression(C=5.0, max_iter=1000, solver="lbfgs",
                                     random_state=42)),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("clf",   LinearSVC(C=1.0, max_iter=2000, random_state=42)),
    ]),
    "LSA + Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf_params)),
        ("svd",   TruncatedSVD(n_components=150, random_state=42)),
        ("clf",   LogisticRegression(C=5.0, max_iter=1000, random_state=42)),
    ]),
}

results = {}
trained_models = {}
print("🏋  Training models…")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted")
    cv_scores = cross_val_score(pipe, X_text, y, cv=cv,
                                scoring="accuracy", n_jobs=-1)
    results[name] = {
        "accuracy": round(acc, 4),
        "f1_weighted": round(f1, 4),
        "cv_mean": round(cv_scores.mean(), 4),
        "cv_std":  round(cv_scores.std(),  4),
        "y_pred":  y_pred.tolist(),
    }
    trained_models[name] = pipe
    print(f"   ✓ {name:<35} acc={acc:.3f}  f1={f1:.3f}  cv={cv_scores.mean():.3f}±{cv_scores.std():.3f}")

# Best model
best_name = max(results, key=lambda k: results[k]["f1_weighted"])
best_model = trained_models[best_name]
print(f"\n🏆 Best model: {best_name}")

# ──────────────────────────────────────────────
# 3. SAVE ARTEFACTS
# ──────────────────────────────────────────────
joblib.dump(best_model, "/home/claude/best_model.pkl")
joblib.dump(le, "/home/claude/label_encoder.pkl")
df.to_csv("/home/claude/dataset.csv", index=False)

# Save results JSON
with open("/home/claude/results.json", "w") as f:
    payload = {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
               for k, v in results.items()}
    json.dump({"results": payload, "best_model": best_name,
               "genres": le.classes_.tolist(),
               "n_samples": len(df),
               "n_train": len(X_train),
               "n_test": len(X_test)}, f, indent=2)

# ──────────────────────────────────────────────
# 4. VISUALISATIONS
# ──────────────────────────────────────────────
DARK_BG  = "#0f0f1a"
ACCENT   = "#e94560"
ACCENT2  = "#00c2ff"
ACCENT3  = "#7fff7f"
TEXT_CLR = "#e8e8f0"
GRID_CLR = "#2a2a3a"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": DARK_BG,
    "text.color": TEXT_CLR, "axes.labelcolor": TEXT_CLR,
    "xtick.color": TEXT_CLR, "ytick.color": TEXT_CLR,
    "axes.edgecolor": GRID_CLR, "grid.color": GRID_CLR,
    "font.family": "monospace",
})

genres_list = le.classes_.tolist()

# ── Fig 1: Model Comparison Bar Chart ──────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Model Performance Comparison", fontsize=16, color=TEXT_CLR,
             fontweight="bold", y=1.02)

metrics = ["accuracy", "f1_weighted", "cv_mean"]
titles  = ["Test Accuracy", "Weighted F1", "Cross-Val Accuracy"]
colors  = [ACCENT, ACCENT2, ACCENT3]
model_names_short = [n.replace(" (Complement)", "\n(Complement)")
                     .replace(" + ", "\n+ ") for n in results.keys()]

for ax, metric, title, color in zip(axes, metrics, titles, colors):
    vals = [results[n][metric] for n in results]
    bars = ax.barh(model_names_short, vals, color=color, alpha=0.85, height=0.55)
    ax.set_xlim(0, 1.0)
    ax.set_title(title, color=color, fontsize=11, pad=8)
    ax.axvline(x=0.5, color=GRID_CLR, linewidth=1, linestyle="--")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=9, color=TEXT_CLR)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("/home/claude/fig_model_comparison.png", dpi=150,
            bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("   ✓ Saved fig_model_comparison.png")

# ── Fig 2: Confusion Matrix for best model ─────
best_preds = results[best_name]["y_pred"]
cm = confusion_matrix(y_test, best_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(11, 9))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
im = ax.imshow(cm_norm, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
cbar = plt.colorbar(im, ax=ax, fraction=0.04)
cbar.ax.yaxis.set_tick_params(color=TEXT_CLR)
cbar.ax.tick_params(labelcolor=TEXT_CLR)

ax.set_xticks(range(len(genres_list)))
ax.set_yticks(range(len(genres_list)))
ax.set_xticklabels(genres_list, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(genres_list, fontsize=9)
ax.set_xlabel("Predicted Genre", labelpad=10)
ax.set_ylabel("True Genre", labelpad=10)
ax.set_title(f"Confusion Matrix — {best_name}", color=ACCENT2,
             fontsize=13, fontweight="bold", pad=14)

for i in range(len(genres_list)):
    for j in range(len(genres_list)):
        clr = "black" if cm_norm[i, j] > 0.6 else TEXT_CLR
        ax.text(j, i, f"{cm[i,j]}", ha="center", va="center",
                fontsize=8, color=clr, fontweight="bold")

plt.tight_layout()
plt.savefig("/home/claude/fig_confusion_matrix.png", dpi=150,
            bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("   ✓ Saved fig_confusion_matrix.png")

# ── Fig 3: Per-genre F1 ─────────────────────────
report = classification_report(y_test, best_preds,
                                target_names=genres_list, output_dict=True)
genre_f1 = [report[g]["f1-score"] for g in genres_list]

fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor(DARK_BG)
palette = plt.cm.plasma(np.linspace(0.2, 0.9, len(genres_list)))
bars = ax.bar(genres_list, genre_f1, color=palette, width=0.65)
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1 Score", fontsize=11)
ax.set_title(f"Per-Genre F1 Score — {best_name}", fontsize=13,
             color=ACCENT, fontweight="bold", pad=12)
ax.axhline(y=np.mean(genre_f1), color=ACCENT2, linestyle="--",
           linewidth=1.5, label=f"Mean = {np.mean(genre_f1):.3f}")
ax.legend(framealpha=0.2, labelcolor=TEXT_CLR)
for bar, val in zip(bars, genre_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, color=TEXT_CLR)
ax.set_xticklabels(genres_list, rotation=35, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("/home/claude/fig_per_genre_f1.png", dpi=150,
            bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("   ✓ Saved fig_per_genre_f1.png")

# ── Fig 4: Dataset distribution ────────────────
genre_counts = df["genre"].value_counts()
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK_BG)
palette2 = plt.cm.cool(np.linspace(0.1, 0.9, len(genre_counts)))
bars = ax.bar(genre_counts.index, genre_counts.values, color=palette2, width=0.65)
ax.set_title("Dataset Distribution by Genre", fontsize=13,
             color=ACCENT3, fontweight="bold", pad=12)
ax.set_ylabel("Sample Count", fontsize=11)
for bar, val in zip(bars, genre_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            str(val), ha="center", fontsize=9, color=TEXT_CLR)
ax.set_xticklabels(genre_counts.index, rotation=35, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("/home/claude/fig_dataset_dist.png", dpi=150,
            bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print("   ✓ Saved fig_dataset_dist.png")

print("\n✅ Training complete! All artefacts saved.")
