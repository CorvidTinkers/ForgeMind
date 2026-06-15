"""
ForgeMind Knowledge Extractor.

Extracts structured knowledge from dataset text:
- Key entities
- Key concepts
- Frequent terms / domain keywords
- Entity relationships (lightweight knowledge graph)

Uses spaCy for NER and dependency parsing when available,
falls back to frequency-based extraction.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Lazy-load spaCy
_nlp = None
_SPACY_AVAILABLE = False

try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    logger.warning("spaCy not installed. Using fallback knowledge extraction.")


def _get_nlp():
    """Lazy-load the spaCy model."""
    global _nlp
    if _nlp is None and _SPACY_AVAILABLE:
        try:
            _nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
    return _nlp


# Common stop words to filter out of keywords
STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "that", "this", "these",
    "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
    "he", "him", "his", "she", "her", "they", "them", "their", "what",
    "which", "who", "whom",
}


class KnowledgeExtractor:
    """
    Extracts entities, concepts, keywords, and relationships
    from dataset text to build a lightweight knowledge graph.
    """

    def extract(
        self,
        sample_texts: list[str],
        max_entities: int = 50,
        max_keywords: int = 30,
        max_relationships: int = 30,
    ) -> dict:
        """
        Extract knowledge from text samples.

        Args:
            sample_texts: List of text samples.
            max_entities: Maximum entities to return.
            max_keywords: Maximum keywords to return.
            max_relationships: Maximum entity relationships to return.

        Returns:
            Dict with 'entities', 'concepts', 'keywords', 'relationships'.
        """
        if not sample_texts:
            return {
                "entities": [],
                "concepts": [],
                "keywords": [],
                "relationships": [],
            }

        # Combine samples (limit to avoid processing too much text)
        combined_text = "\n".join(sample_texts[:200])

        nlp = _get_nlp()
        if nlp is not None:
            return self._extract_with_spacy(
                nlp, combined_text, max_entities, max_keywords, max_relationships
            )
        return self._extract_fallback(
            combined_text, sample_texts, max_entities, max_keywords
        )

    def _extract_with_spacy(
        self,
        nlp,
        text: str,
        max_entities: int,
        max_keywords: int,
        max_relationships: int,
    ) -> dict:
        """Extract knowledge using spaCy NER and dependency parsing."""
        # Process in chunks to handle large texts
        max_chars = 100_000
        doc = nlp(text[:max_chars])

        # Extract named entities
        entity_counter = Counter()
        entity_types = {}
        for ent in doc.ents:
            clean = ent.text.strip()
            if len(clean) > 1:
                entity_counter[clean] += 1
                entity_types[clean] = ent.label_

        entities = [
            ent for ent, _ in entity_counter.most_common(max_entities)
        ]

        # Extract noun chunks as concepts
        concept_counter = Counter()
        for chunk in doc.noun_chunks:
            clean = chunk.text.strip().lower()
            if len(clean) > 2 and clean not in STOP_WORDS:
                concept_counter[clean] += 1

        concepts = [
            concept for concept, _ in concept_counter.most_common(max_keywords)
        ]

        # Extract keywords (frequent meaningful tokens)
        keywords = self._extract_keywords(text, max_keywords)

        # Extract entity relationships
        relationships = self._extract_relationships_spacy(
            doc, entity_types, max_relationships
        )

        return {
            "entities": entities,
            "concepts": concepts,
            "keywords": keywords,
            "relationships": relationships,
        }

    def _extract_relationships_spacy(
        self,
        doc,
        entity_types: dict[str, str],
        max_relationships: int,
    ) -> list[dict]:
        """
        Extract entity relationships from dependency parse.

        Looks for subject-verb-object triples where the subject
        and object are named entities.
        """
        relationships = []
        seen = set()

        for sent in doc.sents:
            subjects = []
            objects = []
            verb = None

            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass"):
                    subjects.append(token.text)
                elif token.dep_ in ("dobj", "pobj", "attr"):
                    objects.append(token.text)
                elif token.pos_ == "VERB" and verb is None:
                    verb = token.lemma_

            if verb and subjects and objects:
                for subj in subjects:
                    for obj in objects:
                        key = (subj.lower(), obj.lower(), verb)
                        if key not in seen:
                            seen.add(key)
                            relationships.append({
                                "source": subj,
                                "target": obj,
                                "relation": verb,
                            })

            if len(relationships) >= max_relationships:
                break

        return relationships[:max_relationships]

    def _extract_fallback(
        self,
        text: str,
        sample_texts: list[str],
        max_entities: int,
        max_keywords: int,
    ) -> dict:
        """Fallback extraction using frequency analysis."""
        keywords = self._extract_keywords(text, max_keywords)

        # Use capitalized words as pseudo-entities
        entity_pattern = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b")
        entity_counter = Counter()
        for match in entity_pattern.finditer(text):
            entity = match.group().strip()
            if len(entity) > 2 and entity.lower() not in STOP_WORDS:
                entity_counter[entity] += 1

        entities = [ent for ent, _ in entity_counter.most_common(max_entities)]

        return {
            "entities": entities,
            "concepts": keywords[:max_keywords // 2],
            "keywords": keywords,
            "relationships": [],
        }

    def _extract_keywords(self, text: str, max_keywords: int) -> list[str]:
        """Extract frequent meaningful words as keywords."""
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        word_counts = Counter(words)

        # Filter stop words
        keywords = [
            word
            for word, count in word_counts.most_common(max_keywords * 3)
            if word not in STOP_WORDS and count >= 2
        ]

        return keywords[:max_keywords]
