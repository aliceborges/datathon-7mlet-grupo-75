import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_telemarketing_data(input_dir: Path, output_dir: Path, files: list) -> None:
    """
    Carrega as bases de treino e teste originais do Kaggle, padroniza os nomes
    das colunas e remove variáveis irrelevantes (como Unnamed: 0) ou com vazamento.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for file_name in files:
        input_file = input_dir / file_name
        output_file = output_dir / f"{input_file.stem}_clean.csv"

        logging.info(f"A processar o ficheiro: {input_file}")

        try:
            df = pd.read_csv(input_file, sep=';')
            df.columns = [col.replace('.', '_') for col in df.columns]
            cols_to_drop = ['id', 'duration', 'Unnamed: 0']
            logging.info(f"A remover as colunas indesejadas (ID e Leakage).")
            df_clean = df.drop(columns=cols_to_drop, errors='ignore')
            df_clean = df_clean.loc[:, ~df_clean.columns.str.contains('^Unnamed')]
            df_clean.to_csv(output_file, index=False)
            logging.info(f"Ficheiro limpo gerado com sucesso em: {output_file}\n")
        except FileNotFoundError:
            logging.error(f"Ficheiro não encontrado: {input_file}. Verifique a pasta.")
        except Exception as e:
            logging.error(f"Erro inesperado no ficheiro {file_name}: {e}")

if __name__ == "__main__":
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    INPUT_DIR = PROJECT_DIR / "data" / "kaggle"
    OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

    FILES_TO_PROCESS = ["train.csv", "test.csv"]

    process_telemarketing_data(INPUT_DIR, OUTPUT_DIR, FILES_TO_PROCESS)