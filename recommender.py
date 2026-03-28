from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pickle
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier


DEFAULT_DATA_DIR = Path(r"C:\Users\navis\OneDrive\Desktop\Data Science2")
DEFAULT_HISTORY_PATH = DEFAULT_DATA_DIR / "history.csv"
DEFAULT_INVESTORS_PATH = DEFAULT_DATA_DIR / "investors.csv"
DEFAULT_INVENTORS_PATH = DEFAULT_DATA_DIR / "inventors.csv"
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
AUGMENTED_HISTORY_PATH = PROJECT_DATA_DIR / "history_augmented.csv"
MODEL_CACHE_PATH = PROJECT_DATA_DIR / "model_cache.pkl"
CACHE_VERSION = 5
AUGMENTATION_VERSION = 2
MATCH_THRESHOLD = 3

RISK_MAP = {
    "Very Low": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
    "Very High": 5,
}


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    confusion: Dict[str, int] = field(default_factory=dict)


class PlatformRecommender:
    numeric_features = [
        "investor_id_numeric",
        "idea_id_numeric",
        "available_funds",
        "company_investment",
        "past_investments",
        "funding_required",
        "team_size",
        "text_similarity",
        "domain_match",
        "location_match",
        "risk_gap",
        "affordability_ratio",
        "capital_strength",
        "heuristic_score",
        "investor_positive_rate",
        "investor_avg_score",
        "investor_interaction_count",
        "idea_positive_rate",
        "idea_avg_score",
        "idea_interaction_count",
        "domain_positive_rate",
        "technology_positive_rate",
        "location_positive_rate",
    ]
    categorical_features = [
        "investor_id_bucket",
        "idea_id_bucket",
        "preferred_risk_appetite",
        "preferred_location",
        "industry_focus",
        "focus_domain",
        "domain",
        "technology",
        "risk_level",
        "location",
    ]

    def __init__(
        self,
        history_path: Path | str = AUGMENTED_HISTORY_PATH,
        investors_path: Path | str = DEFAULT_INVESTORS_PATH,
        inventors_path: Path | str = DEFAULT_INVENTORS_PATH,
        random_state: int = 42,
    ) -> None:
        self.history_path = Path(history_path)
        self.investors_path = Path(investors_path)
        self.inventors_path = Path(inventors_path)
        self.random_state = random_state

        self.history_df: pd.DataFrame | None = None
        self.raw_history_df: pd.DataFrame | None = None
        self.investors_df: pd.DataFrame | None = None
        self.inventors_df: pd.DataFrame | None = None
        self.positive_pairs: set[Tuple[int, int]] = set()

        self.investor_text_index: Dict[int, int] = {}
        self.inventor_text_index: Dict[int, int] = {}
        self.investor_tfidf = None
        self.inventor_tfidf = None

        self.models: Dict[str, ModelResult] = {}
        self.best_model_name: str | None = None
        self.investor_stats: pd.DataFrame | None = None
        self.idea_stats: pd.DataFrame | None = None
        self.domain_stats: pd.DataFrame | None = None
        self.tech_stats: pd.DataFrame | None = None
        self.location_stats: pd.DataFrame | None = None
        self.investor_recommendation_cache: Dict[Tuple[int, int], List[Dict]] = {}
        self.inventor_match_cache: Dict[Tuple[int, int], List[Dict]] = {}
        self._augmentation_ready = False

    def train(self) -> None:
        self._ensure_augmented_history()
        self._load_data()
        self._prepare_text_features()
        self.investor_recommendation_cache.clear()
        self.inventor_match_cache.clear()

        self._build_aggregate_stats()
        if self._load_model_cache():
            return
        observed_rows = self._build_observed_rows(self.history_df)
        train_rows, test_rows = train_test_split(
            observed_rows,
            test_size=0.25,
            stratify=observed_rows["label"],
            random_state=self.random_state,
        )

        synthetic_rows = self._build_synthetic_training_rows()
        if synthetic_rows.empty or self._augmentation_ready:
            synthetic_rows = pd.DataFrame(columns=["investor_id", "idea_id", "label", "weight"])
        else:
            train_rows = pd.concat([train_rows, synthetic_rows], ignore_index=True)

        X_train = self.build_pair_features(train_rows[["investor_id", "idea_id"]])
        y_train = train_rows["label"].astype(int)
        w_train = train_rows["weight"].astype(float)

        X_test = self.build_pair_features(test_rows[["investor_id", "idea_id"]])
        y_test = test_rows["label"].astype(int)

        model_specs = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                max_depth=16,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=220,
                learning_rate=0.05,
                max_depth=3,
                random_state=self.random_state,
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=20,
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=self.random_state,
            ),
        }

        self.models = {}
        for name, estimator in model_specs.items():
            pipeline = self._build_pipeline(estimator)
            pipeline.fit(X_train, y_train, model__sample_weight=w_train)

            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

            metrics = {
                "accuracy": float(accuracy_score(y_test, predictions)),
                "precision": float(precision_score(y_test, predictions, zero_division=0)),
                "recall": float(recall_score(y_test, predictions, zero_division=0)),
                "f1": float(f1_score(y_test, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, probabilities)),
            }
            metrics["overall"] = float(
                0.2 * metrics["accuracy"]
                + 0.2 * metrics["precision"]
                + 0.2 * metrics["recall"]
                + 0.2 * metrics["f1"]
                + 0.2 * metrics["roc_auc"]
            )
            self.models[name] = ModelResult(
                name=name,
                pipeline=pipeline,
                metrics=metrics,
                confusion={"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            )

        self.best_model_name = max(
            self.models.values(),
            key=lambda item: item.metrics["overall"],
        ).name

        full_rows = observed_rows
        if not synthetic_rows.empty:
            full_rows = pd.concat([observed_rows, synthetic_rows], ignore_index=True)
        X_full = self.build_pair_features(full_rows[["investor_id", "idea_id"]])
        y_full = full_rows["label"].astype(int)
        w_full = full_rows["weight"].astype(float)

        for model_result in self.models.values():
            model_result.pipeline.fit(X_full, y_full, model__sample_weight=w_full)
        self._save_model_cache()

    def _ensure_augmented_history(self) -> None:
        if Path(self.history_path).exists() and Path(self.history_path) != AUGMENTED_HISTORY_PATH:
            return

        PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)

        raw_history = pd.read_csv(DEFAULT_HISTORY_PATH)
        investors = pd.read_csv(DEFAULT_INVESTORS_PATH)
        inventors = pd.read_csv(DEFAULT_INVENTORS_PATH)

        raw_history["interaction_score"] = raw_history["interaction_score"].astype(int)
        synthetic_rows = self._build_synthetic_history(raw_history, investors, inventors)
        augmented = pd.concat([raw_history, synthetic_rows], ignore_index=True)
        augmented.to_csv(self.history_path, index=False)
        self._augmentation_ready = True

    def _load_model_cache(self) -> bool:
        if not MODEL_CACHE_PATH.exists():
            return False
        try:
            with MODEL_CACHE_PATH.open("rb") as handle:
                payload = pickle.load(handle)
            if payload.get("version") != CACHE_VERSION:
                return False
            if payload.get("augmentation_version") != AUGMENTATION_VERSION:
                return False
            self.models = payload["models"]
            self.best_model_name = payload["best_model_name"]
            return True
        except Exception:
            return False

    def _save_model_cache(self) -> None:
        PROJECT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "augmentation_version": AUGMENTATION_VERSION,
            "models": self.models,
            "best_model_name": self.best_model_name,
        }
        with MODEL_CACHE_PATH.open("wb") as handle:
            pickle.dump(payload, handle)

    def _build_pipeline(self, estimator) -> Pipeline:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_features),
                ("cat", categorical_pipeline, self.categorical_features),
            ]
        )
        return Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])

    def _load_data(self) -> None:
        self.raw_history_df = pd.read_csv(DEFAULT_HISTORY_PATH)
        self.history_df = self.raw_history_df.copy()
        self.investors_df = pd.read_csv(self.investors_path)
        self.inventors_df = pd.read_csv(self.inventors_path)

        self.history_df["investor_id"] = self.history_df["investor_id"].astype(int)
        self.history_df["idea_id"] = self.history_df["idea_id"].astype(int)
        self.history_df["interaction_score"] = self.history_df["interaction_score"].astype(int)

        self.investors_df["investor_id"] = self.investors_df["investor_id"].astype(int)
        self.investors_df["available_funds"] = self.investors_df["available_funds"].astype(float)
        self.investors_df["company_investment"] = self.investors_df["company_investment"].astype(float)
        self.investors_df["past_investments"] = self.investors_df["past_investments"].astype(float)
        self.investors_df["risk_numeric"] = self.investors_df["preferred_risk_appetite"].map(RISK_MAP).fillna(3)
        self.investors_df["profile_text"] = (
            self.investors_df["focus_domain"].fillna("")
            + " "
            + self.investors_df["industry_focus"].fillna("")
            + " "
            + self.investors_df["preferred_location"].fillna("")
            + " "
            + self.investors_df["preferred_risk_appetite"].fillna("")
        )

        self.inventors_df["idea_id"] = self.inventors_df["idea_id"].astype(int)
        self.inventors_df["funding_st_required"] = self.inventors_df["funding_st_required"].astype(float)
        self.inventors_df["f_team_size"] = self.inventors_df["f_team_size"].astype(float)
        self.inventors_df["risk_numeric"] = self.inventors_df["risk_level"].map(RISK_MAP).fillna(3)
        self.inventors_df["profile_text"] = (
            self.inventors_df["idea_title"].fillna("")
            + " "
            + self.inventors_df["idea_text"].fillna("")
            + " "
            + self.inventors_df["domain"].fillna("")
            + " "
            + self.inventors_df["technology"].fillna("")
            + " "
            + self.inventors_df["location"].fillna("")
            + " "
            + self.inventors_df["risk_level"].fillna("")
        )

        self.positive_pairs = {
            (int(row.investor_id), int(row.idea_id))
            for row in self.history_df.itertuples()
            if int(row.interaction_score) > 0
        }
        self._build_aggregate_stats()

    def _build_aggregate_stats(self) -> None:
        stats_frame = self.history_df.copy()
        stats_frame["positive_flag"] = (stats_frame["interaction_score"] >= MATCH_THRESHOLD).astype(int)
        stats_frame = stats_frame.merge(
            self.investors_df[["investor_id", "focus_domain", "preferred_location"]],
            on="investor_id",
            how="left",
        ).merge(
            self.inventors_df[["idea_id", "domain", "technology", "location"]],
            on="idea_id",
            how="left",
        )

        self.investor_stats = (
            stats_frame.groupby("investor_id")
            .agg(
                investor_positive_rate=("positive_flag", "mean"),
                investor_avg_score=("interaction_score", "mean"),
                investor_interaction_count=("interaction_score", "count"),
            )
            .reset_index()
        )
        self.idea_stats = (
            stats_frame.groupby("idea_id")
            .agg(
                idea_positive_rate=("positive_flag", "mean"),
                idea_avg_score=("interaction_score", "mean"),
                idea_interaction_count=("interaction_score", "count"),
            )
            .reset_index()
        )
        self.domain_stats = (
            stats_frame.groupby("domain")
            .agg(domain_positive_rate=("positive_flag", "mean"))
            .reset_index()
        )
        self.tech_stats = (
            stats_frame.groupby("technology")
            .agg(technology_positive_rate=("positive_flag", "mean"))
            .reset_index()
        )
        self.location_stats = (
            stats_frame.groupby("location")
            .agg(location_positive_rate=("positive_flag", "mean"))
            .reset_index()
        )

    @staticmethod
    def _token_overlap(left: str, right: str) -> float:
        left_tokens = {token for token in left.lower().split() if len(token) > 2}
        right_tokens = {token for token in right.lower().split() if len(token) > 2}
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)

    def _raw_heuristic_score(self, investor_row: pd.Series, idea_row: pd.Series) -> float:
        risk_gap = abs(RISK_MAP.get(str(investor_row.get("preferred_risk_appetite", "")).strip(), 3) - RISK_MAP.get(str(idea_row.get("risk_level", "")).strip(), 3))
        domain_match = int(str(investor_row.get("focus_domain", "")).strip().lower() == str(idea_row.get("domain", "")).strip().lower())
        location_match = int(str(investor_row.get("preferred_location", "")).strip().lower() == str(idea_row.get("location", "")).strip().lower())
        text_overlap = self._token_overlap(
            f"{investor_row.get('focus_domain', '')} {investor_row.get('industry_focus', '')}",
            f"{idea_row.get('idea_title', '')} {idea_row.get('idea_text', '')} {idea_row.get('domain', '')} {idea_row.get('technology', '')}",
        )
        funding_required = float(idea_row.get("funding_st_required", 1) or 1)
        available_funds = float(investor_row.get("available_funds", 0) or 0)
        affordability = min(available_funds / max(funding_required, 1.0), 3.0) / 3.0
        company_inv = float(investor_row.get("company_investment", 0) or 0)
        capital_strength = min((available_funds + company_inv) / max(funding_required, 1.0), 4.0) / 4.0

        return float(
            2.6 * domain_match
            + 2.0 * location_match
            + 1.7 * (1.0 - min(risk_gap, 5) / 5.0)
            + 1.8 * text_overlap
            + 1.6 * affordability
            + 0.8 * capital_strength
        )

    def _build_synthetic_history(
        self,
        raw_history: pd.DataFrame,
        investors: pd.DataFrame,
        inventors: pd.DataFrame,
        per_investor_candidates: int = 30,
        positives_per_investor: int = 10,
        negatives_per_investor: int = 10,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        rows = []
        next_history_id = int(raw_history["history_id"].max()) + 1 if not raw_history.empty else 1
        investor_map = investors.set_index("investor_id").to_dict(orient="index")
        idea_map = inventors.set_index("idea_id").to_dict(orient="index")
        idea_domain_map = (
            inventors.groupby(inventors["domain"].fillna("").str.lower())["idea_id"]
            .apply(lambda s: s.astype(int).tolist())
            .to_dict()
        )
        idea_location_map = (
            inventors.groupby(inventors["location"].fillna("").str.lower())["idea_id"]
            .apply(lambda s: s.astype(int).tolist())
            .to_dict()
        )
        idea_risk_map = (
            inventors.groupby(inventors["risk_level"].fillna("").str.lower())["idea_id"]
            .apply(lambda s: s.astype(int).tolist())
            .to_dict()
        )

        for investor_id in investors["investor_id"].astype(int).tolist():
            investor_row = investor_map[int(investor_id)]
            investor_series = pd.Series(investor_row)
            focus_domain = str(investor_row.get("focus_domain", "")).strip().lower()
            preferred_location = str(investor_row.get("preferred_location", "")).strip().lower()
            investor_risk = str(investor_row.get("preferred_risk_appetite", "")).strip().lower()

            positive_pool = set(idea_domain_map.get(focus_domain, []))
            positive_pool.update(idea_location_map.get(preferred_location, []))
            positive_pool.update(idea_risk_map.get(investor_risk, []))

            positive_pool = [idea_id for idea_id in positive_pool if (int(investor_id), int(idea_id)) not in self.positive_pairs]
            if not positive_pool:
                positive_pool = [int(idea_id) for idea_id in inventors["idea_id"].astype(int).tolist()]

            negative_pool = inventors[
                inventors["domain"].fillna("").str.lower().ne(focus_domain)
                & inventors["location"].fillna("").str.lower().ne(preferred_location)
                & inventors["risk_level"].fillna("").str.lower().ne(investor_risk)
            ]["idea_id"].astype(int).tolist()
            negative_pool = [idea_id for idea_id in negative_pool if (int(investor_id), int(idea_id)) not in self.positive_pairs]
            if not negative_pool:
                negative_pool = [int(idea_id) for idea_id in inventors["idea_id"].astype(int).tolist()]

            neutral_pool = [
                int(idea_id)
                for idea_id in inventors["idea_id"].astype(int).tolist()
                if idea_id not in positive_pool and idea_id not in negative_pool
            ]
            if not neutral_pool:
                neutral_pool = [int(idea_id) for idea_id in inventors["idea_id"].astype(int).tolist()]

            candidate_ideas = []
            for pool, size in (
                (positive_pool, max(8, per_investor_candidates // 3)),
                (negative_pool, max(8, per_investor_candidates // 3)),
                (neutral_pool, max(4, per_investor_candidates - 2 * max(8, per_investor_candidates // 3))),
            ):
                if not pool:
                    continue
                chosen = rng.choice(pool, size=min(size, len(pool)), replace=False)
                candidate_ideas.extend([int(item) for item in chosen.tolist()])

            candidate_ideas = list(dict.fromkeys(candidate_ideas))
            scored_rows = []
            for idea_id in candidate_ideas:
                idea_row = idea_map[int(idea_id)]
                score = self._raw_heuristic_score(investor_series, pd.Series(idea_row))
                scored_rows.append((idea_id, score))
            scored_rows.sort(key=lambda x: x[1], reverse=True)

            for idea_id, score in scored_rows[:positives_per_investor]:
                if score >= 6.2:
                    interaction_score = 5
                elif score >= 5.2:
                    interaction_score = 4
                else:
                    interaction_score = 3
                rows.append(
                    {
                        "history_id": next_history_id,
                        "investor_id": int(investor_id),
                        "idea_id": int(idea_id),
                        "year": int(2015 + (score * 3) % 9),
                        "interaction_score": interaction_score,
                    }
                )
                next_history_id += 1

            for idea_id, score in scored_rows[-negatives_per_investor:]:
                if score <= 2.8:
                    interaction_score = 0
                elif score <= 3.6:
                    interaction_score = 1
                else:
                    interaction_score = 2
                rows.append(
                    {
                        "history_id": next_history_id,
                        "investor_id": int(investor_id),
                        "idea_id": int(idea_id),
                        "year": int(2015 + (score * 4) % 9),
                        "interaction_score": interaction_score,
                    }
                )
                next_history_id += 1

        return pd.DataFrame(rows)

    def _append_synthetic_training_rows(self) -> None:
        rows = []
        rng = np.random.default_rng(self.random_state)
        for investor_id in self.investors_df["investor_id"].astype(int).tolist():
            candidate_ideas = rng.choice(
                self.inventors_df["idea_id"].astype(int).tolist(),
                size=min(20, len(self.inventors_df)),
                replace=False,
            )
            pair_frame = pd.DataFrame(
                {
                    "investor_id": [investor_id] * len(candidate_ideas),
                    "idea_id": candidate_ideas.tolist(),
                }
            )
            features = self.build_pair_features(pair_frame)
            scored = features.join(pair_frame)
            scored["heuristic_score"] = (
                2.5 * scored["domain_match"]
                + 2.0 * scored["location_match"]
                + 1.8 * (1.0 - (scored["risk_gap"] / 5.0).clip(0, 1))
                + 1.8 * scored["text_similarity"]
                + 1.4 * (scored["affordability_ratio"] / 2.0).clip(0, 1.5)
                + 0.8 * scored["investor_positive_rate"]
                + 0.8 * scored["idea_positive_rate"]
                + 0.6 * scored["domain_positive_rate"].fillna(0)
                + 0.6 * scored["technology_positive_rate"].fillna(0)
                + 0.6 * scored["location_positive_rate"].fillna(0)
            )
            top_hits = scored.sort_values("heuristic_score", ascending=False).head(5)
            bottom_hits = scored.sort_values("heuristic_score", ascending=True).head(5)

            for _, row in top_hits.iterrows():
                rows.append(
                    {
                        "investor_id": int(row["investor_id"]),
                        "idea_id": int(row["idea_id"]),
                        "interaction_score": 5,
                        "label": 1,
                        "weight": 0.8,
                    }
                )

            for _, row in bottom_hits.iterrows():
                rows.append(
                    {
                        "investor_id": int(row["investor_id"]),
                        "idea_id": int(row["idea_id"]),
                        "interaction_score": 0,
                        "label": 0,
                        "weight": 0.8,
                    }
                )

        synthetic_frame = pd.DataFrame(rows)
        if synthetic_frame.empty:
            return

        extra_history = synthetic_frame.assign(
            history_id=np.arange(
                int(self.history_df["history_id"].max()) + 1,
                int(self.history_df["history_id"].max()) + 1 + len(synthetic_frame),
            ),
            year=2024,
        )[["history_id", "investor_id", "idea_id", "year", "interaction_score"]]

        self.history_df = pd.concat([self.history_df, extra_history], ignore_index=True)
        self.history_df = self.history_df.drop_duplicates(subset=["investor_id", "idea_id", "year"], keep="last")

    def _prepare_text_features(self) -> None:
        combined_text = pd.concat(
            [
                self.investors_df["profile_text"].fillna(""),
                self.inventors_df["profile_text"].fillna(""),
            ],
            ignore_index=True,
        )
        vectorizer = TfidfVectorizer(stop_words="english", max_features=600)
        vectorizer.fit(combined_text)

        self.investor_tfidf = vectorizer.transform(self.investors_df["profile_text"].fillna(""))
        self.inventor_tfidf = vectorizer.transform(self.inventors_df["profile_text"].fillna(""))
        self.investor_text_index = {
            int(investor_id): idx for idx, investor_id in enumerate(self.investors_df["investor_id"])
        }
        self.inventor_text_index = {
            int(idea_id): idx for idx, idea_id in enumerate(self.inventors_df["idea_id"])
        }

    def _sample_unobserved_negatives(self, count: int) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        investors = self.investors_df["investor_id"].tolist()
        ideas = self.inventors_df["idea_id"].tolist()

        sampled_pairs = set()
        results: List[Tuple[int, int]] = []
        max_attempts = max(count * 10, 1000)
        attempts = 0
        while len(results) < count and attempts < max_attempts:
            investor_id = int(rng.choice(investors))
            idea_id = int(rng.choice(ideas))
            pair = (investor_id, idea_id)
            if pair in self.positive_pairs or pair in sampled_pairs:
                attempts += 1
                continue
            sampled_pairs.add(pair)
            results.append(pair)
            attempts += 1

        frame = pd.DataFrame(results, columns=["investor_id", "idea_id"])
        frame["label"] = 0
        frame["weight"] = 1.0
        return frame

    def _build_observed_rows(self, history_df: pd.DataFrame) -> pd.DataFrame:
        rows = history_df[["investor_id", "idea_id", "interaction_score"]].copy()
        rows["label"] = (rows["interaction_score"] >= MATCH_THRESHOLD).astype(int)
        rows["weight"] = np.where(
            rows["label"] == 1,
            1.0 + (rows["interaction_score"] - (MATCH_THRESHOLD - 1)) / 5.0,
            1.0,
        )
        return rows[["investor_id", "idea_id", "label", "weight"]]

    def _build_synthetic_training_rows(self) -> pd.DataFrame:
        synthetic_history = self._build_synthetic_history(self.history_df, self.investors_df, self.inventors_df)
        if synthetic_history.empty:
            return pd.DataFrame(columns=["investor_id", "idea_id", "label", "weight"])
        synthetic_rows = synthetic_history[["investor_id", "idea_id", "interaction_score"]].copy()
        synthetic_rows["label"] = (synthetic_rows["interaction_score"] >= MATCH_THRESHOLD).astype(int)
        synthetic_rows["weight"] = np.where(synthetic_rows["label"] == 1, 0.85, 0.45)
        return synthetic_rows[["investor_id", "idea_id", "label", "weight"]]

    def _pair_text_similarity(self, investor_ids: pd.Series, idea_ids: pd.Series) -> np.ndarray:
        values = []
        for investor_id, idea_id in zip(investor_ids, idea_ids):
            inv_idx = self.investor_text_index.get(int(investor_id))
            idea_idx = self.inventor_text_index.get(int(idea_id))
            if inv_idx is None or idea_idx is None:
                values.append(0.0)
                continue
            score = self.investor_tfidf[inv_idx].multiply(self.inventor_tfidf[idea_idx]).sum()
            values.append(float(score))
        return np.array(values)

    def build_pair_features(self, pairs: pd.DataFrame) -> pd.DataFrame:
        merged = (
            pairs.merge(self.investors_df, on="investor_id", how="left")
            .merge(self.inventors_df, on="idea_id", how="left", suffixes=("_investor", "_inventor"))
            .copy()
        )
        merged = merged.merge(self.investor_stats, on="investor_id", how="left")
        merged = merged.merge(self.idea_stats, on="idea_id", how="left")
        merged = merged.merge(self.domain_stats, on="domain", how="left")
        merged = merged.merge(self.tech_stats, on="technology", how="left")
        merged = merged.merge(self.location_stats, on="location", how="left")

        merged["text_similarity"] = self._pair_text_similarity(merged["investor_id"], merged["idea_id"])
        merged["domain_match"] = (
            merged["focus_domain"].fillna("").str.lower()
            == merged["domain"].fillna("").str.lower()
        ).astype(int)
        merged["location_match"] = (
            merged["preferred_location"].fillna("").str.lower()
            == merged["location"].fillna("").str.lower()
        ).astype(int)
        merged["risk_gap"] = (
            merged["risk_numeric_investor"].fillna(3) - merged["risk_numeric_inventor"].fillna(3)
        ).abs()
        merged["affordability_ratio"] = (
            merged["available_funds"].fillna(0) / merged["funding_st_required"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0).clip(0, 10)
        merged["capital_strength"] = (
            merged["company_investment"].fillna(0) + merged["available_funds"].fillna(0)
        ) / (merged["funding_st_required"].fillna(1) + 1)
        merged["heuristic_score"] = (
            2.5 * merged["domain_match"]
            + 2.0 * merged["location_match"]
            + 1.8 * (1.0 - (merged["risk_gap"] / 5.0).clip(0, 1))
            + 1.8 * merged["text_similarity"]
            + 1.4 * (merged["affordability_ratio"] / 2.0).clip(0, 1.5)
            + 0.8 * merged["investor_positive_rate"].fillna(0)
            + 0.8 * merged["idea_positive_rate"].fillna(0)
            + 0.6 * merged["domain_positive_rate"].fillna(0)
            + 0.6 * merged["technology_positive_rate"].fillna(0)
            + 0.6 * merged["location_positive_rate"].fillna(0)
        )
        merged["investor_id_numeric"] = merged["investor_id"].astype(float)
        merged["idea_id_numeric"] = merged["idea_id"].astype(float)
        merged["investor_id_bucket"] = "INV_" + merged["investor_id"].astype(str)
        merged["idea_id_bucket"] = "IDEA_" + merged["idea_id"].astype(str)

        renamed = merged.rename(
            columns={
                "funding_st_required": "funding_required",
                "f_team_size": "team_size",
            }
        )
        return renamed[self.numeric_features + self.categorical_features]

    def get_model_comparison(self) -> List[Dict[str, float]]:
        ordered = sorted(
            (
                result.metrics | {"name": result.name} | {"confusion": result.confusion}
                for result in self.models.values()
            ),
            key=lambda row: row["overall"],
            reverse=True,
        )
        return ordered

    def get_graph_payload(self) -> Dict:
        comparison = self.get_model_comparison()
        return {
            "labels": [row["name"] for row in comparison],
            "accuracy": [round(row["accuracy"] * 100, 2) for row in comparison],
            "precision": [round(row["precision"] * 100, 2) for row in comparison],
            "recall": [round(row["recall"] * 100, 2) for row in comparison],
            "f1": [round(row["f1"] * 100, 2) for row in comparison],
            "roc_auc": [round(row["roc_auc"] * 100, 2) for row in comparison],
            "confusion": [
                {"name": row["name"], **row["confusion"]} for row in comparison
            ],
        }

    def get_market_insights(self) -> Dict:
        top_domains = (
            self.inventors_df["domain"].value_counts().head(5).reset_index().to_dict(orient="records")
        )
        top_technologies = (
            self.inventors_df["technology"].value_counts().head(5).reset_index().to_dict(orient="records")
        )
        return {
            "top_domains": [
                {"label": row["domain"], "value": int(row["count"])} for row in top_domains
            ],
            "top_technologies": [
                {"label": row["technology"], "value": int(row["count"])} for row in top_technologies
            ],
            "positive_interactions": int((self.history_df["interaction_score"] >= MATCH_THRESHOLD).sum()),
            "catalog_size": int(len(self.inventors_df)),
        }

    def _score_pairs(self, pairs: pd.DataFrame) -> Dict[str, np.ndarray]:
        features = self.build_pair_features(pairs)
        return {
            name: result.pipeline.predict_proba(features)[:, 1]
            for name, result in self.models.items()
        }

    def _recommendation_reason(self, raw_row: pd.Series) -> List[str]:
        reasons = []
        if raw_row["domain_match"] == 1:
            reasons.append("domain focus matches exactly")
        if raw_row["location_match"] == 1:
            reasons.append("location preference aligns")
        if raw_row["text_similarity"] >= 0.1:
            reasons.append("strong text similarity between investor interests and invention")
        if raw_row["affordability_ratio"] >= 1:
            reasons.append("investor can comfortably fund this ask")
        if raw_row["risk_gap"] <= 1:
            reasons.append("risk appetite is compatible")
        if not reasons:
            reasons.append("overall pattern matches the strongest historical signals")
        return reasons[:3]

    def recommend_for_investor(self, investor_id: int, top_n: int = 8) -> List[Dict]:
        cache_key = (int(investor_id), int(top_n))
        if cache_key in self.investor_recommendation_cache:
            return self.investor_recommendation_cache[cache_key]

        candidate_pairs = pd.DataFrame(
            {
                "investor_id": [investor_id] * len(self.inventors_df),
                "idea_id": self.inventors_df["idea_id"].tolist(),
            }
        )
        seen_ideas = {
            idea_id for inv_id, idea_id in self.positive_pairs if inv_id == int(investor_id)
        }
        candidate_pairs = candidate_pairs[~candidate_pairs["idea_id"].isin(seen_ideas)].reset_index(drop=True)
        scored = self._score_pairs(candidate_pairs)
        feature_rows = self.build_pair_features(candidate_pairs).reset_index(drop=True)

        result = candidate_pairs.copy()
        for model_name, scores in scored.items():
            result[model_name] = scores
        result["best_score"] = result[self.best_model_name]

        enriched = result.merge(self.inventors_df, on="idea_id", how="left")
        enriched = pd.concat(
            [
                enriched.reset_index(drop=True),
                feature_rows[
                    [
                        "text_similarity",
                        "domain_match",
                        "location_match",
                        "risk_gap",
                        "affordability_ratio",
                    ]
                ],
            ],
            axis=1,
        )
        enriched = enriched.sort_values("best_score", ascending=False).head(top_n)

        recommendations = []
        for row in enriched.to_dict(orient="records"):
            recommendations.append(
                {
                    "idea_id": int(row["idea_id"]),
                    "idea_title": row["idea_title"],
                    "domain": row["domain"],
                    "technology": row["technology"],
                    "risk_level": row["risk_level"],
                    "location": row["location"],
                    "funding_required": float(row["funding_st_required"]),
                    "team_size": int(row["f_team_size"]),
                    "summary": row["idea_text"],
                    "best_model": self.best_model_name,
                    "best_score": round(float(row["best_score"]) * 100, 2),
                    "model_scores": {
                        name: round(float(row[name]) * 100, 2) for name in self.models.keys()
                    },
                    "reasons": self._recommendation_reason(pd.Series(row)),
                }
            )
        self.investor_recommendation_cache[cache_key] = recommendations
        return recommendations

    def recommend_for_inventor(self, idea_id: int, top_n: int = 8) -> List[Dict]:
        cache_key = (int(idea_id), int(top_n))
        if cache_key in self.inventor_match_cache:
            return self.inventor_match_cache[cache_key]

        candidate_pairs = pd.DataFrame(
            {
                "investor_id": self.investors_df["investor_id"].tolist(),
                "idea_id": [idea_id] * len(self.investors_df),
            }
        )
        scored = self._score_pairs(candidate_pairs)
        feature_rows = self.build_pair_features(candidate_pairs).reset_index(drop=True)

        result = candidate_pairs.copy()
        for model_name, scores in scored.items():
            result[model_name] = scores
        result["best_score"] = result[self.best_model_name]

        enriched = result.merge(self.investors_df, on="investor_id", how="left")
        enriched = pd.concat(
            [
                enriched.reset_index(drop=True),
                feature_rows[
                    [
                        "text_similarity",
                        "domain_match",
                        "location_match",
                        "risk_gap",
                        "affordability_ratio",
                    ]
                ],
            ],
            axis=1,
        )
        enriched = enriched.sort_values("best_score", ascending=False).head(top_n)

        matches = []
        for row in enriched.to_dict(orient="records"):
            matches.append(
                {
                    "investor_id": int(row["investor_id"]),
                    "investor_name": row["investor_name"],
                    "focus_domain": row["focus_domain"],
                    "industry_focus": row["industry_focus"],
                    "preferred_location": row["preferred_location"],
                    "preferred_risk_appetite": row["preferred_risk_appetite"],
                    "available_funds": float(row["available_funds"]),
                    "best_model": self.best_model_name,
                    "best_score": round(float(row["best_score"]) * 100, 2),
                    "model_scores": {
                        name: round(float(row[name]) * 100, 2) for name in self.models.keys()
                    },
                    "reasons": self._recommendation_reason(pd.Series(row)),
                }
            )
        self.inventor_match_cache[cache_key] = matches
        return matches

    def get_investor(self, investor_id: int) -> Dict | None:
        row = self.investors_df[self.investors_df["investor_id"] == investor_id]
        if row.empty:
            return None
        record = row.iloc[0].to_dict()
        record["available_funds"] = float(record["available_funds"])
        record["company_investment"] = float(record["company_investment"])
        record["past_investments"] = int(record["past_investments"])
        return record

    def get_inventor(self, idea_id: int) -> Dict | None:
        row = self.inventors_df[self.inventors_df["idea_id"] == idea_id]
        if row.empty:
            return None
        record = row.iloc[0].to_dict()
        record["funding_st_required"] = float(record["funding_st_required"])
        record["f_team_size"] = int(record["f_team_size"])
        return record
