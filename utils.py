import json
import logging
from dataclasses import dataclass
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuração dos Testes ---
BASE_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = BASE_DIR / 'fixtures' / 'summarization_data'

@dataclass
class EvaluationItem:
    """Classe de dados para representar um item de avaliação."""
    id: str
    ai_summary: str
    reference_summary: str
    expected_keywords: list[str] | None = None
    max_summary_length_words: int | None = None
    min_summary_length_words: int | None = None
    ai_feeling: str | None = None
    reference_feeling: str | None = None

def load_evaluation_data(filename: str = "evaluation_data.json") -> list[dict]:
    """
    Carrega dados de avaliação do arquivo JSON.
    
    Args:
        filename: Nome do arquivo JSON contendo os dados de avaliação
        
    Returns:
        Lista de dicionários contendo os dados de avaliação
        
    Raises:
        FileNotFoundError: Se o arquivo de dados não for encontrado
        json.JSONDecodeError: Se o arquivo JSON estiver mal formatado
    """
    file_path = FIXTURES_DIR / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo de dados de avaliação não encontrado: {file_path}")

    with file_path.open(encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError("Os dados de avaliação devem ser uma lista de itens")

    return data 