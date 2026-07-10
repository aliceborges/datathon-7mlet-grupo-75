from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SYNTHETIC_SEED = 75
DEFAULT_HORIZON_DAYS = 14
DEFAULT_CATALOG_SIZE = 12
DEFAULT_OUTPUT_DIRNAME = "generated"

_FALLBACK_JOBS = [
    "admin.",
    "technician",
    "services",
    "management",
    "blue-collar",
    "retired",
]
_FALLBACK_CONTACTS = ["cellular", "telephone"]
_OFFER_TEMPLATES = [
    {"offer_type": "cashback", "reward_type": "cashback", "reward_value": 25.0},
    {"offer_type": "voucher", "reward_type": "voucher", "reward_value": 18.0},
    {"offer_type": "fee_waiver", "reward_type": "fee_waiver", "reward_value": 12.0},
    {"offer_type": "bonus_points", "reward_type": "bonus_points", "reward_value": 35.0},
]


@dataclass(frozen=True)
class SyntheticEnrichmentBundle:
    offer_catalog: pd.DataFrame
    offer_events: pd.DataFrame
    delayed_rewards: pd.DataFrame


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized.columns = [col.replace(".", "_") for col in normalized.columns]
    return normalized


def _coerce_source_frame(source_df: pd.DataFrame) -> pd.DataFrame:
    if source_df.empty:
        raise ValueError("source_df nao pode estar vazio")
    return _normalize_columns(source_df).reset_index(drop=True)


def _get_column_values(df: pd.DataFrame, column: str, fallback: list[str]) -> list[str]:
    if column not in df.columns:
        return fallback.copy()
    values = [str(value) for value in df[column].dropna().astype(str).unique().tolist()]
    return values or fallback.copy()


def _get_age_band(age: int) -> str:
    if age < 30:
        return "18-29"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def _age_bounds_for_band(age_band: str) -> tuple[int, int]:
    if age_band == "18-29":
        return 18, 29
    if age_band == "30-44":
        return 30, 44
    if age_band == "45-59":
        return 45, 59
    return 60, 80


def _pick_offer_index(offer_catalog: pd.DataFrame, row: pd.Series, rng: np.random.Generator) -> int:
    age = int(row.get("age", 40))
    job = str(row.get("job", "unknown"))
    contact = str(row.get("contact", "unknown"))
    weights: list[float] = []
    for _, offer in offer_catalog.iterrows():
        score = 1.0
        if int(offer["target_age_min"]) <= age <= int(offer["target_age_max"]):
            score += 1.8
        if job == str(offer["target_job"]):
            score += 0.9
        if contact == str(offer["channel"]):
            score += 0.7
        score += float(offer["expected_acceptance_rate"]) * 2.0
        weights.append(max(score, 0.05))
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    return int(rng.choice(len(offer_catalog), p=probabilities))


def _build_offer_catalog(
    source_df: pd.DataFrame,
    rng: np.random.Generator,
    catalog_size: int,
    start_timestamp: pd.Timestamp,
    horizon_days: int,
) -> pd.DataFrame:
    jobs = _get_column_values(source_df, "job", _FALLBACK_JOBS)
    contacts = _get_column_values(source_df, "contact", _FALLBACK_CONTACTS)
    catalog_rows: list[dict[str, object]] = []
    for index in range(catalog_size):
        template = _OFFER_TEMPLATES[index % len(_OFFER_TEMPLATES)]
        age_band = ["18-29", "30-44", "45-59", "60+"][index % 4]
        min_age, max_age = _age_bounds_for_band(age_band)
        offer_id = f"offer_{index + 1:03d}"
        target_job = jobs[index % len(jobs)]
        channel = contacts[index % len(contacts)]
        reward_value = float(template["reward_value"] + rng.integers(0, 6))
        discount_rate = round(0.05 + 0.02 * (index % 4) + float(rng.random()) * 0.01, 3)
        expected_acceptance_rate = round(0.08 + 0.03 * (index % 4) + float(rng.random()) * 0.02, 3)
        expected_reward_delay_days = int(1 + (index % min(horizon_days, 7)))
        catalog_rows.append(
            {
                "offer_id": offer_id,
                "offer_type": template["offer_type"],
                "channel": channel,
                "reward_type": template["reward_type"],
                "reward_value": reward_value,
                "discount_rate": discount_rate,
                "target_job": target_job,
                "target_age_min": min_age,
                "target_age_max": max_age,
                "valid_from": start_timestamp + pd.Timedelta(days=index),
                "valid_until": start_timestamp + pd.Timedelta(days=index + horizon_days),
                "expected_acceptance_rate": expected_acceptance_rate,
                "expected_reward_delay_days": expected_reward_delay_days,
            }
        )
    return pd.DataFrame(catalog_rows)


def _build_offer_events(
    source_df: pd.DataFrame,
    offer_catalog: pd.DataFrame,
    rng: np.random.Generator,
    n_events: int,
    start_timestamp: pd.Timestamp,
    horizon_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sampled_rows = rng.choice(source_df.index.to_numpy(), size=n_events, replace=True)
    event_rows: list[dict[str, object]] = []
    reward_rows: list[dict[str, object]] = []
    reward_columns = [
        "reward_id",
        "event_id",
        "client_id",
        "offer_id",
        "reward_timestamp",
        "delay_days",
        "reward_value",
        "reward_type",
        "reward_reason",
    ]

    for event_index, source_row_id in enumerate(sampled_rows, start=1):
        row = source_df.iloc[int(source_row_id)]
        age = int(row.get("age", int(rng.integers(18, 70))))
        age_band = _get_age_band(age)
        job = str(row.get("job", "unknown"))
        marital = str(row.get("marital", "unknown"))
        education = str(row.get("education", "unknown"))
        contact = str(row.get("contact", "unknown"))
        source_y = str(row.get("y", "unknown")).lower()

        offer_index = _pick_offer_index(offer_catalog, row, rng)
        offer = offer_catalog.iloc[offer_index]

        event_timestamp = start_timestamp + pd.Timedelta(
            days=int(rng.integers(0, horizon_days)),
            hours=int(rng.integers(8, 20)),
            minutes=int(rng.integers(0, 60)),
        )

        propensity_score = float(offer["expected_acceptance_rate"])
        if int(offer["target_age_min"]) <= age <= int(offer["target_age_max"]):
            propensity_score += 0.08
        if job == str(offer["target_job"]):
            propensity_score += 0.06
        if contact == str(offer["channel"]):
            propensity_score += 0.04
        if source_y == "yes":
            propensity_score += 0.05
        elif source_y == "no":
            propensity_score -= 0.02
        if int(row.get("campaign", 0)) > 1:
            propensity_score -= 0.01 * min(int(row.get("campaign", 0)) - 1, 4)
        propensity_score = float(np.clip(propensity_score, 0.03, 0.95))

        accepted_offer = bool(rng.random() < propensity_score)
        converted_with_delay = bool(
            accepted_offer and (rng.random() < min(0.85, propensity_score + 0.18))
        )
        max_delay = max(1, min(7, horizon_days - int((event_timestamp - start_timestamp).days)))
        reward_delay_days = int(rng.integers(1, max_delay + 1)) if converted_with_delay else pd.NA

        event_rows.append(
            {
                "event_id": f"event_{event_index:05d}",
                "source_row_id": int(source_row_id),
                "client_id": f"client_{event_index:05d}",
                "offer_id": str(offer["offer_id"]),
                "offer_type": str(offer["offer_type"]),
                "channel": str(offer["channel"]),
                "event_timestamp": event_timestamp,
                "age": age,
                "age_band": age_band,
                "job": job,
                "marital": marital,
                "education": education,
                "contact": contact,
                "source_y": source_y,
                "propensity_score": round(propensity_score, 4),
                "accepted_offer": accepted_offer,
                "converted_with_delay": converted_with_delay,
                "reward_delay_days": reward_delay_days,
            }
        )

        if converted_with_delay:
            reward_rows.append(
                {
                    "reward_id": f"reward_{event_index:05d}",
                    "event_id": f"event_{event_index:05d}",
                    "client_id": f"client_{event_index:05d}",
                    "offer_id": str(offer["offer_id"]),
                    "reward_timestamp": event_timestamp + pd.Timedelta(days=reward_delay_days),
                    "delay_days": reward_delay_days,
                    "reward_value": float(offer["reward_value"]),
                    "reward_type": str(offer["reward_type"]),
                    "reward_reason": "delayed_conversion",
                }
            )

    events_df = pd.DataFrame(event_rows)
    rewards_df = pd.DataFrame(reward_rows, columns=reward_columns)
    events_df = events_df.astype({"reward_delay_days": "Int64"})
    return events_df, rewards_df


def generate_synthetic_enrichment(
    source_df: pd.DataFrame,
    seed: int = DEFAULT_SYNTHETIC_SEED,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    catalog_size: int = DEFAULT_CATALOG_SIZE,
    n_events: int | None = None,
    start_timestamp: str | pd.Timestamp = "2024-01-01",
) -> SyntheticEnrichmentBundle:
    """Gera a camada sintetica com catalogo, eventos e recompensas atrasadas."""

    prepared_df = _coerce_source_frame(source_df)
    if n_events is None:
        n_events = max(len(prepared_df) * 2, catalog_size * 10)

    rng = np.random.default_rng(seed)
    start_ts = pd.Timestamp(start_timestamp)
    offer_catalog = _build_offer_catalog(prepared_df, rng, catalog_size, start_ts, horizon_days)
    offer_events, delayed_rewards = _build_offer_events(
        prepared_df,
        offer_catalog,
        rng,
        n_events,
        start_ts,
        horizon_days,
    )
    return SyntheticEnrichmentBundle(
        offer_catalog=offer_catalog,
        offer_events=offer_events,
        delayed_rewards=delayed_rewards,
    )


def write_synthetic_enrichment(
    bundle: SyntheticEnrichmentBundle,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "offer_catalog": output_dir / "offer_catalog.csv",
        "offer_events": output_dir / "offer_events.csv",
        "delayed_rewards": output_dir / "delayed_rewards.csv",
    }
    bundle.offer_catalog.to_csv(outputs["offer_catalog"], index=False)
    bundle.offer_events.to_csv(outputs["offer_events"], index=False)
    bundle.delayed_rewards.to_csv(outputs["delayed_rewards"], index=False)
    return outputs


def _load_source_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de origem nao encontrado: {path}")
    return pd.read_csv(path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gera a camada sintetica de experimentacao adaptativa.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/train_clean.csv"),
        help="CSV de base para inferir distribuicoes e segmentos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_enrichment") / DEFAULT_OUTPUT_DIRNAME,
        help="Diretorio de saida para offer_catalog, offer_events e delayed_rewards.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--horizon-days", type=int, default=DEFAULT_HORIZON_DAYS)
    parser.add_argument("--catalog-size", type=int, default=DEFAULT_CATALOG_SIZE)
    parser.add_argument("--n-events", type=int, default=None)
    parser.add_argument(
        "--start-timestamp",
        type=str,
        default="2024-01-01",
        help="Timestamp inicial da simulacao.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    source_df = _load_source_csv(args.input)
    bundle = generate_synthetic_enrichment(
        source_df=source_df,
        seed=args.seed,
        horizon_days=args.horizon_days,
        catalog_size=args.catalog_size,
        n_events=args.n_events,
        start_timestamp=args.start_timestamp,
    )
    outputs = write_synthetic_enrichment(bundle, args.output)
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())