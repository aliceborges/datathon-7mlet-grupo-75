import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

EXPECTED_FEATURE_COLUMNS = [
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
TARGET_COLUMN = "y"
DROP_COLUMNS = ["id", "duration", "Unnamed: 0"]


def _expected_columns(file_name: str) -> list[str]:
    if file_name == "train.csv":
        return EXPECTED_FEATURE_COLUMNS + [TARGET_COLUMN]
    if file_name == "test.csv":
        return EXPECTED_FEATURE_COLUMNS.copy()
    raise ValueError(f"Arquivo nao suportado: {file_name}")


def _validate_processed_schema(df: pd.DataFrame, file_name: str) -> None:
    expected_columns = _expected_columns(file_name)
    actual_columns = df.columns.tolist()
    if actual_columns != expected_columns:
        raise ValueError(
            "Schema processado inesperado para "
            f"{file_name}: esperado {expected_columns}, encontrado {actual_columns}"
        )


def process_telemarketing_data(input_dir: Path, output_dir: Path, files: list) -> None:
    """
    Carrega as bases de treino e teste originais do Kaggle, padroniza os nomes
    das colunas, remove identificadores e valida o schema da camada processada.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in files:
        input_file = input_dir / file_name
        output_file = output_dir / f"{input_file.stem}_clean.csv"

        logging.info(f"A processar o ficheiro: {input_file}")

        try:
            df = pd.read_csv(input_file, sep=';')
            df.columns = [col.replace('.', '_') for col in df.columns]
            logging.info("A remover colunas de identificacao e vazamento temporal.")
            df_clean = df.drop(columns=DROP_COLUMNS, errors='ignore')
            df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
            _validate_processed_schema(df_clean, file_name)
            df_clean.to_csv(output_file, index=False)
            logging.info(f"Ficheiro limpo gerado com sucesso em: {output_file}\n")
        except FileNotFoundError:
            logging.error(f"Ficheiro não encontrado: {input_file}. Verifique a pasta.")
        except ValueError as e:
            logging.error(f"Schema invalido no ficheiro {file_name}: {e}")
            raise
        except Exception as e:
            logging.error(f"Erro inesperado no ficheiro {file_name}: {e}")

if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    INPUT_DIR = PROJECT_DIR / "data" / "kaggle"
    OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

    FILES_TO_PROCESS = ["train.csv", "test.csv"]

    process_telemarketing_data(INPUT_DIR, OUTPUT_DIR, FILES_TO_PROCESS)