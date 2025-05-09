import json
import os
import unittest

import google.generativeai as genai
from dotenv import load_dotenv

from utils import load_evaluation_data, logger

# Carregar variáveis de ambiente
load_dotenv()

# Configurar Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY não encontrada nas variáveis de ambiente")

genai.configure(api_key=GEMINI_API_KEY)

class GeminiConfig:
    """Configurações para o modelo Gemini."""
    
    def __init__(
        self,
        temperature: float = 0.1,  # Baixa temperatura para respostas mais determinísticas
        top_k: int = 1,  # Considera apenas a melhor resposta
        top_p: float = 0.8,  # Probabilidade acumulada para seleção de tokens
        candidate_count: int = 1,  # Número de respostas a gerar
        max_output_tokens: int = 1024,  # Limite de tokens na resposta
        safety_settings: dict | None = None  # Configurações de segurança
    ):
        """
        Inicializa as configurações do Gemini.
        
        Args:
            temperature: Controla a aleatoriedade (0.0 a 1.0)
            top_k: Número de melhores tokens a considerar
            top_p: Probabilidade acumulada para seleção de tokens
            candidate_count: Número de respostas a gerar
            max_output_tokens: Limite de tokens na resposta
            safety_settings: Configurações de segurança
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.candidate_count = candidate_count
        self.max_output_tokens = max_output_tokens
        self.safety_settings = safety_settings or {
            "HARASSMENT": "block_none",
            "HATE_SPEECH": "block_none",
            "SEXUALLY_EXPLICIT": "block_none",
            "DANGEROUS_CONTENT": "block_none"
        }

    def to_generation_config(self) -> dict:
        """Converte as configurações para o formato esperado pelo Gemini."""
        return {
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "candidate_count": self.candidate_count,
            "max_output_tokens": self.max_output_tokens,
            "safety_settings": self.safety_settings
        }

class TestGeminiSummaryComparison(unittest.TestCase):
    """Suite de testes para comparação de resumos usando o Gemini."""

    @classmethod
    def setUpClass(cls) -> None:
        """
        Configura a classe de teste carregando dados e inicializando o modelo Gemini.
        
        Este método é chamado uma vez antes de todos os testes da classe.
        """
        logger.info("Configurando classe de teste Gemini")
        cls.evaluation_data = load_evaluation_data()
        
        # Configurações padrão para testes de similaridade
        cls.similarity_config = GeminiConfig(
            temperature=0.1,  # Baixa temperatura para respostas mais consistentes
            top_k=1,
            top_p=0.8,
            max_output_tokens=1024
        )
        
        # Configurações para verificação de palavras-chave
        cls.keyword_config = GeminiConfig(
            temperature=0.0,  # Temperatura zero para respostas mais precisas
            top_k=1,
            top_p=1.0,
            max_output_tokens=512
        )
        
        cls.model = genai.GenerativeModel('gemini-2.0-flash')

    def _get_gemini_comparison(self, summary1: str, summary2: str, feeling1: str, feeling2: str) -> dict[str, float]:
        """
        Obtém a comparação entre dois resumos usando o Gemini.
        
        Args:
            summary1: Primeiro resumo para comparação
            summary2: Segundo resumo para comparação
            feeling1: Sentimento do primeiro resumo
            feeling2: Sentimento do segundo resumo
            
        Returns:
            Dicionário contendo a pontuação de similaridade e explicação
        """
        prompt = f"""
        Analise os dois resumos abaixo e avalie o quão similares eles são em termos de:
        1. Conteúdo principal
        2. Informações-chave
        3. Estrutura e organização
        4. Tom e estilo
        5. Sentimento geral (feeling)

        Resumo 1: {summary1}
        Sentimento 1: {feeling1}

        Resumo 2: {summary2}
        Sentimento 2: {feeling2}

        IMPORTANTE: Responda APENAS com um JSON no seguinte formato, sem nenhum texto adicional:
        {{"score": 0.XX, "explanation": "explicação detalhada"}}
        """

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.similarity_config.to_generation_config()
            )
            response_text = response.text.strip()
            
            # Tenta encontrar o JSON na resposta
            try:
                # Primeira tentativa: resposta direta
                return json.loads(response_text)
            except json.JSONDecodeError:
                # Segunda tentativa: procurar JSON entre chaves
                json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
                if not json_str:
                    raise ValueError("Nenhum JSON encontrado na resposta")
                return json.loads(json_str)
                
        except Exception:
            logger.exception("Erro ao obter comparação do Gemini")
            return {"score": 0.0, "explanation": "Erro na comparação"}

    def test_gemini_summary_similarity(self) -> None:
        """Testa a similaridade entre resumos usando o Gemini."""
        logger.info("Executando testes de similaridade com Gemini")

        for item in self.evaluation_data:
            ai_summary = item['ai_summary']
            reference_summary = item['reference_summary']
            ai_feeling = item.get('ai_feeling', '')
            reference_feeling = item.get('reference_feeling', '')

            with self.subTest(eval_id=item['id']):
                logger.info(f"Avaliando item: {item['id']}")

                # Validar entradas
                self.assertTrue(ai_summary, f"Resumo da IA está vazio para {item['id']}")
                self.assertTrue(reference_summary, f"Resumo de referência está vazio para {item['id']}")

                # Obter comparação do Gemini
                comparison = self._get_gemini_comparison(
                    ai_summary, 
                    reference_summary,
                    ai_feeling,
                    reference_feeling
                )
                logger.info(f"Comparação Gemini para {item['id']}: {comparison}")

                # Verificar se a pontuação está acima do limiar
                self.assertGreaterEqual(
                    comparison['score'],
                    0.6,
                    f"Similaridade baixa para {item['id']} (Esperado >= 0.6, Obtido: {comparison['score']:.2f})\n"
                    f"Explicação: {comparison['explanation']}"
                )

    def test_gemini_feeling_match(self) -> None:
        """Testa se os sentimentos (feelings) dos resumos são compatíveis."""
        logger.info("Executando testes de compatibilidade de sentimentos")

        for item in self.evaluation_data:
            if 'ai_feeling' not in item or 'reference_feeling' not in item:
                continue

            ai_feeling = item['ai_feeling']
            reference_feeling = item['reference_feeling']

            with self.subTest(eval_id=item['id']):
                logger.info(f"Avaliando sentimentos para item: {item['id']}")

                # Verifica se os sentimentos são iguais
                self.assertEqual(
                    ai_feeling.lower(),
                    reference_feeling.lower(),
                    f"Sentimentos incompatíveis para {item['id']}: "
                    f"AI feeling '{ai_feeling}' != Reference feeling '{reference_feeling}'"
                )

    def test_gemini_keyword_presence(self) -> None:
        """Testa se o Gemini identifica a presença de palavras-chave e seus sinônimos no resumo."""
        logger.info("Executando testes de presença de palavras-chave com Gemini")

        for item in self.evaluation_data:
            if "expected_keywords" not in item or not item["expected_keywords"]:
                continue

            ai_summary = item['ai_summary']
            expected_keywords = item['expected_keywords']

            with self.subTest(eval_id=item['id']):
                logger.info(f"Avaliando item: {item['id']}")

                prompt = f"""
                Analise o resumo abaixo e verifique se ele contém as seguintes palavras-chave ou seus sinônimos/conceitos equivalentes.
                Considere variações linguísticas, termos médicos equivalentes e expressões similares.

                Palavras-chave a verificar:
                {', '.join(expected_keywords)}

                Exemplos de equivalência:
                - "dor de cabeça" = "enxaqueca" = "cefaleia" = "dor na cabeça"
                - "sensibilidade à luz" = "fotofobia" = "incomodo com luz"
                - "problema de sono" = "insônia" = "dificuldade para dormir" = "má qualidade do sono"
                - "falta de refeições" = "pular refeições" = "não se alimentar" = "jejum prolongado"

                Resumo: {ai_summary}

                IMPORTANTE: Responda APENAS com um JSON no seguinte formato, sem nenhum texto adicional:
                {{
                    "found": ["palavra1", "palavra2"],
                    "missing": ["palavra3", "palavra4"],
                    "equivalences": {{
                        "palavra1": ["sinonimo1", "sinonimo2"],
                        "palavra2": ["sinonimo1", "sinonimo2"]
                    }}
                }}
                """

                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config=self.keyword_config.to_generation_config()
                    )
                    response_text = response.text.strip()
                    
                    # Tenta encontrar o JSON na resposta
                    try:
                        # Primeira tentativa: resposta direta
                        result = json.loads(response_text)
                    except json.JSONDecodeError:
                        # Segunda tentativa: procurar JSON entre chaves
                        json_str = response_text[response_text.find("{"):response_text.rfind("}")+1]
                        if not json_str:
                            raise ValueError("Nenhum JSON encontrado na resposta")
                        result = json.loads(json_str)

                    # Verificar se todas as palavras-chave foram encontradas
                    self.assertEqual(
                        len(result['missing']),
                        0,
                        f"Palavras-chave não encontradas no resumo para {item['id']}: {result['missing']}\n"
                        f"Equivalências encontradas: {result['equivalences']}\n"
                        f"Resumo: '{ai_summary}'"
                    )
                except Exception:
                    logger.exception("Erro ao verificar palavras-chave com Gemini")
                    self.fail(f"Falha ao verificar palavras-chave para {item['id']}")

if __name__ == '__main__':
    unittest.main(verbosity=2) 