from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.make_dataset import process_telemarketing_data

TRAIN_HEADER = (
    "id;age;job;marital;education;default;housing;loan;contact;month;"
    "day_of_week;campaign;pdays;previous;poutcome;emp.var.rate;cons.price.idx;"
    "cons.conf.idx;euribor3m;nr.employed;y\n"
)
TEST_HEADER = (
    "id;age;job;marital;education;default;housing;loan;contact;month;"
    "day_of_week;campaign;pdays;previous;poutcome;emp.var.rate;cons.price.idx;"
    "cons.conf.idx;euribor3m;nr.employed\n"
)


def _write_raw_csv(path: Path, header: str, row: str) -> None:
    path.write_text(header + row, encoding="utf-8")


def test_process_telemarketing_data_enforces_processed_schema(tmp_path: Path):
    input_dir = tmp_path / "kaggle"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()

    _write_raw_csv(
        input_dir / "train.csv",
        TRAIN_HEADER,
        "1;30;admin.;married;university.degree;no;yes;no;cellular;jul;mon;1;999;0;failure;1.1;93.994;-36.4;4.857;5191.0;yes\n",
    )
    _write_raw_csv(
        input_dir / "test.csv",
        TEST_HEADER,
        "2;41;services;single;high.school;no;no;yes;telephone;aug;tue;2;6;1;success;-0.1;92.893;-46.2;1.313;5099.1\n",
    )

    process_telemarketing_data(input_dir, output_dir, ["train.csv", "test.csv"])

    train_clean = pd.read_csv(output_dir / "train_clean.csv")
    test_clean = pd.read_csv(output_dir / "test_clean.csv")

    assert train_clean.columns.tolist() == [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp_var_rate",
        "cons_price_idx",
        "cons_conf_idx",
        "euribor3m",
        "nr_employed",
        "y",
    ]
    assert test_clean.columns.tolist() == [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp_var_rate",
        "cons_price_idx",
        "cons_conf_idx",
        "euribor3m",
        "nr_employed",
    ]
    assert "id" not in train_clean.columns
    assert "duration" not in train_clean.columns
    assert "duration" not in test_clean.columns


def test_process_telemarketing_data_rejects_train_without_target(tmp_path: Path):
    input_dir = tmp_path / "kaggle"
    output_dir = tmp_path / "processed"
    input_dir.mkdir()

    _write_raw_csv(
        input_dir / "train.csv",
        TEST_HEADER,
        "1;30;admin.;married;university.degree;no;yes;no;cellular;jul;mon;1;999;0;failure;1.1;93.994;-36.4;4.857;5191.0\n",
    )

    with pytest.raises(ValueError, match="Schema processado inesperado"):
        process_telemarketing_data(input_dir, output_dir, ["train.csv"])
