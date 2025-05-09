import re
import unittest

import nltk
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, AutoTokenizer

from utils import load_evaluation_data, logger


class PortugueseTextProcessor:
    """Classe para processamento de texto em português."""

    def __init__(self):
        """Inicializa o processador de texto com os recursos necessários."""
        self._setup_nltk_resources()
        self._setup_bert()

    def _setup_nltk_resources(self):
        """Baixa os recursos necessários do NLTK para processamento em português."""
        resources = [
            ('tokenizers/punkt', 'punkt'),  # Para tokenização de sentenças
        ]

        for resource_path, resource_name in resources:
            try:
                nltk.data.find(resource_path)
                logger.debug(f"Recurso NLTK {resource_name} já baixado")
            except LookupError:
                logger.info(f"Baixando recurso NLTK: {resource_name}")
                nltk.download(resource_name, quiet=True)

    def _setup_bert(self):
        """Configura o modelo BERTimbau para embeddings em português."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
            self.model = AutoModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
            self.model.eval()  # Modo de avaliação
            logger.info("Modelo BERTimbau carregado com sucesso")
        except Exception:
            logger.exception("Erro ao carregar modelo BERTimbau")
            raise

    def get_bert_embedding(self, text: str) -> np.ndarray:
        """
        Gera embeddings usando o BERTimbau.
        
        Args:
            text: Texto de entrada em português
            
        Returns:
            Array numpy com o embedding do texto
        """
        try:
            # Tokeniza o texto
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

            # Gera embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Usa o embedding da última camada do [CLS]
                embeddings = outputs.last_hidden_state[:, 0, :].numpy()

            return embeddings[0]  # Retorna o embedding do primeiro (e único) texto
        except Exception:
            logger.exception("Erro ao gerar embedding BERT")
            return np.zeros(768)  # Retorna vetor zero em caso de erro

    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade semântica entre dois textos usando BERTimbau.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Score de similaridade entre 0 e 1
        """
        if not text1 or not text2:
            return 0.0

        # Gera embeddings
        emb1 = self.get_bert_embedding(text1)
        emb2 = self.get_bert_embedding(text2)

        # Calcula similaridade de cosseno
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(similarity)

    def preprocess_text(self, text: str) -> str:
        """
        Pré-processa texto em português removendo caracteres especiais e normalizando.
        
        Args:
            text: Texto de entrada em português
            
        Returns:
            Texto pré-processado
        """
        # Remove acentos
        text = self._remove_accents(text)
        # Remove caracteres especiais mantendo pontuação básica
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
        # Normaliza espaços
        text = ' '.join(text.split())
        return text.lower()

    def _remove_accents(self, text: str) -> str:
        """Remove acentos do texto em português."""
        import unicodedata
        return ''.join(c for c in unicodedata.normalize('NFD', text)
                      if unicodedata.category(c) != 'Mn')

    def tokenize(self, text: str) -> list[str]:
        """
        Tokeniza texto em português.
        
        Args:
            text: Texto de entrada em português
            
        Returns:
            Lista de tokens
        """
        return word_tokenize(self.preprocess_text(text), language='portuguese')

class TestSummaryEvaluation(unittest.TestCase):
    """Suite de testes para avaliação de resumos gerados por IA."""

    @classmethod
    def setUpClass(cls) -> None:
        """
        Configura a classe de teste carregando dados e inicializando o processador de texto.
        
        Este método é chamado uma vez antes de todos os testes da classe.
        """
        logger.info("Configurando classe de teste")
        cls.evaluation_data = load_evaluation_data()
        cls.text_processor = PortugueseTextProcessor()
        cls.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        cls.smooth = SmoothingFunction().method1
        cls.vectorizer = TfidfVectorizer(
            tokenizer=cls.text_processor.tokenize,
            stop_words='english',
            ngram_range=(1, 2)
        )

    def _calculate_rouge(self, hypothesis: str, reference: str) -> dict[str, float]:
        """
        Calcula métricas ROUGE entre dois textos.
        
        Args:
            hypothesis: Texto gerado
            reference: Texto de referência
            
        Returns:
            Dicionário com scores ROUGE
        """
        scores = self.rouge_scorer.score(reference, hypothesis)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }

    def _calculate_bleu(self, hypothesis: str, reference: str) -> float:
        """
        Calcula score BLEU entre dois textos.
        
        Args:
            hypothesis: Texto gerado
            reference: Texto de referência
            
        Returns:
            Score BLEU
        """
        # Pré-processa os textos
        hyp_processed = self.text_processor.preprocess_text(hypothesis)
        ref_processed = self.text_processor.preprocess_text(reference)

        # Tokeniza os textos
        hyp_tokens = self.text_processor.tokenize(hyp_processed)
        ref_tokens = self.text_processor.tokenize(ref_processed)

        # Usa smoothing mais agressivo para lidar com português
        weights = (0.4, 0.3, 0.2, 0.1)  # Dá mais peso para unigramas e bigramas
        return sentence_bleu([ref_tokens], hyp_tokens,
                           weights=weights,
                           smoothing_function=self.smooth)

    def _calculate_cosine_similarity(self, text1: str, text2: str) -> float:
        """
        Calcula similaridade de cosseno entre dois textos.
        
        Args:
            text1: Primeiro texto
            text2: Segundo texto
            
        Returns:
            Score de similaridade entre 0 e 1
        """
        return self.text_processor.calculate_semantic_similarity(text1, text2)

    def _check_keywords(self, text: str, keywords: list[str]) -> list[str]:
        """
        Verifica quais palavras-chave estão presentes no texto.
        
        Args:
            text: Texto para verificar
            keywords: Lista de palavras-chave esperadas
            
        Returns:
            Lista de palavras-chave não encontradas
        """
        if not text or not keywords:
            return keywords if keywords else []

        # Pré-processa o texto e as palavras-chave
        text_processed = self.text_processor.preprocess_text(text)

        missing = []
        for keyword in keywords:
            # Pré-processa a palavra-chave
            keyword_processed = self.text_processor.preprocess_text(keyword)
            # Verifica se a palavra-chave está no texto
            if keyword_processed not in text_processed:
                missing.append(keyword)

        return missing

    def test_summary_quality_metrics(self) -> None:
        """Test various quality metrics for AI-generated summaries."""
        logger.info("Executing summary quality metrics tests")

        for item in self.evaluation_data:
            ai_summary = item['ai_summary']
            reference_summary = item['reference_summary']

            with self.subTest(eval_id=item['id']):
                logger.info(f"Evaluating item: {item['id']}")

                # Validate inputs
                self.assertTrue(ai_summary, f"AI summary is empty for {item['id']}")
                self.assertTrue(reference_summary, f"Reference summary is empty for {item['id']}")

                # ROUGE Test
                rouge_scores = self._calculate_rouge(ai_summary, reference_summary)
                logger.info(f"ROUGE scores for {item['id']}: {rouge_scores}")

                self.assertGreaterEqual(rouge_scores['rouge1'], 0.55,
                    f"Low ROUGE-1 for {item['id']} (Expected >= 0.55, Got: {rouge_scores['rouge1']:.2f})")
                self.assertGreaterEqual(rouge_scores['rougeL'], 0.45,
                    f"Low ROUGE-L for {item['id']} (Expected >= 0.45, Got: {rouge_scores['rougeL']:.2f})")

                # BLEU Test
                bleu_score = self._calculate_bleu(ai_summary, reference_summary)
                logger.info(f"BLEU score for {item['id']}: {bleu_score:.2f}")
                self.assertGreaterEqual(bleu_score, 0.35,
                    f"Low BLEU score for {item['id']} (Expected >= 0.35, Got: {bleu_score:.2f})")

                # Semantic Similarity Test (BERTimbau)
                semantic_sim = self._calculate_cosine_similarity(ai_summary, reference_summary)
                logger.info(f"Semantic similarity for {item['id']}: {semantic_sim:.2f}")
                self.assertGreaterEqual(semantic_sim, 0.75,
                    f"Low Semantic Similarity for {item['id']} (Expected >= 0.75, Got: {semantic_sim:.2f})")

    def test_expected_information_presence_in_ai_summary(self) -> None:
        """Test if AI summaries contain expected keywords."""
        logger.info("Executing expected information presence tests")

        for item in self.evaluation_data:
            if "expected_keywords" not in item or not item["expected_keywords"]:
                continue

            ai_summary = item['ai_summary']
            expected_keywords = item['expected_keywords']

            with self.subTest(eval_id=item['id']):
                logger.info(f"Evaluating item: {item['id']}")
                self.assertTrue(ai_summary, f"AI summary is empty for {item['id']}")

                missing = self._check_keywords(ai_summary, expected_keywords)
                self.assertEqual(len(missing), 0,
                    f"Expected keywords missing in AI summary for {item['id']}: {missing}. Summary: '{ai_summary}'")

    def test_ai_summary_length(self) -> None:
        """Test if AI summaries meet length requirements."""
        logger.info("Executing summary length tests")

        for item in self.evaluation_data:
            if "max_summary_length_words" not in item and "min_summary_length_words" not in item:
                continue

            ai_summary = item['ai_summary']
            self.assertTrue(ai_summary, f"AI summary is empty for {item['id']}")
            num_words = len(nltk.word_tokenize(ai_summary))

            with self.subTest(eval_id=item['id']):
                logger.info(f"Evaluating item: {item['id']} (Length: {num_words} words)")

                if "max_summary_length_words" in item:
                    max_len = item['max_summary_length_words']
                    self.assertLessEqual(num_words, max_len,
                        f"AI summary too long for {item['id']}. Expected <= {max_len}, got {num_words}. Summary: '{ai_summary}'")

                if "min_summary_length_words" in item:
                    min_len = item['min_summary_length_words']
                    self.assertGreaterEqual(num_words, min_len,
                        f"AI summary too short for {item['id']}. Expected >= {min_len}, got {num_words}. Summary: '{ai_summary}'")

if __name__ == '__main__':
    unittest.main(verbosity=2) 