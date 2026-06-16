"""
ForgeMind Instruction Sanity Validator.

Uses spaCy NLP to detect grammatically malformed instructions that
small LLMs frequently hallucinate during synthetic generation.

Common failure modes caught:
  - "Who is quantum computing?"  (Who + inanimate concept)
  - "Who are computers?"         (Who + inanimate plural)
  - "AI has become more advanced" (declarative, not an instruction)
  - Empty or trivially short instructions

This validator runs BEFORE the embedding-based QualityValidator
so that linguistically broken samples never reach the semantic
similarity and diversity scoring stages.
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    import spacy

    try:
        _nlp = spacy.load("en_core_web_sm")
    except OSError:
        _nlp = None
        logger.warning(
            "spaCy model 'en_core_web_sm' not found. "
            "InstructionSanityValidator will use rule-based fallback."
        )
except ImportError:
    spacy = None
    _nlp = None
    logger.warning(
        "spaCy not installed. InstructionSanityValidator will use rule-based fallback."
    )


# Entity labels that are valid targets for "Who" questions.
_WHO_VALID_LABELS = {"PERSON", "ORG", "NORP", "GPE"}

# Minimum instruction length (characters) to be considered valid.
_MIN_INSTRUCTION_LENGTH = 8

# Maximum allowed ratio of repeated words (catches garbled output).
_MAX_WORD_REPEAT_RATIO = 0.6


@dataclass
class SanityResult:
    """Result for a single sample's sanity check."""

    is_valid: bool = True
    reason: str = ""


@dataclass
class SanityReport:
    """Aggregate report from the sanity validation pass."""

    total_checked: int = 0
    passed: int = 0
    failed: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    rejected_examples: list[dict] = field(default_factory=list)


class InstructionSanityValidator:
    """
    NLP-powered sanity checker for generated instructions.

    Checks:
      1. Minimum length
      2. Interrogative-subject agreement ("Who" must target a person/org)
      3. Must be interrogative or imperative (not purely declarative)
      4. No excessive word repetition (garbled LLM output)
    """

    # How many rejected examples to keep in the report.
    MAX_REJECTED_EXAMPLES = 10

    def validate(
        self,
        samples: list[dict],
        instruction_key: str = "instruction",
    ) -> tuple[list[dict], list[dict], SanityReport]:
        """
        Validate a list of samples for instruction sanity.

        Args:
            samples: Generated samples (each a dict).
            instruction_key: The dict key containing the instruction text.

        Returns:
            Tuple of (valid_samples, rejected_samples, SanityReport).
        """
        valid: list[dict] = []
        rejected: list[dict] = []
        reasons: dict[str, int] = {}
        rejected_examples: list[dict] = []

        for sample in samples:
            instruction = self._get_instruction_text(sample, instruction_key)

            if not instruction:
                # No instruction field found; check other text fields
                result = self._check_generic_text(sample)
                logger.debug(
                    "Sanity check (generic): sample_keys=%s, is_valid=%s, reason=%s",
                    list(sample.keys()), result.is_valid, result.reason,
                )
            else:
                result = self._check_instruction(instruction)
                logger.debug(
                    "Sanity check: instruction=%r, is_valid=%s, reason=%s",
                    instruction[:80], result.is_valid, result.reason,
                )

            if result.is_valid:
                valid.append(sample)
            else:
                rejected.append(sample)
                reasons[result.reason] = reasons.get(result.reason, 0) + 1
                if len(rejected_examples) < self.MAX_REJECTED_EXAMPLES:
                    rejected_examples.append({
                        "sample": {
                            k: v for k, v in sample.items()
                            if isinstance(v, (str, int, float, bool))
                        },
                        "reason": result.reason,
                    })

        report = SanityReport(
            total_checked=len(samples),
            passed=len(valid),
            failed=len(rejected),
            rejection_reasons=reasons,
            rejected_examples=rejected_examples,
        )

        logger.info(
            "Sanity validation: checked=%d, passed=%d, failed=%d, reasons=%s",
            report.total_checked,
            report.passed,
            report.failed,
            reasons,
        )

        return valid, rejected, report

    # ------------------------------------------------------------------
    # Core checks
    # ------------------------------------------------------------------

    def _check_instruction(self, text: str) -> SanityResult:
        """Run all checks on a single instruction string."""

        # Check 1: Minimum length
        if len(text.strip()) < _MIN_INSTRUCTION_LENGTH:
            logger.debug("REJECT (too_short): %r", text[:80])
            return SanityResult(is_valid=False, reason="too_short")

        # Check 2: Excessive word repetition
        if self._has_excessive_repetition(text):
            logger.debug("REJECT (garbled_repetition): %r", text[:80])
            return SanityResult(is_valid=False, reason="garbled_repetition")

        # Check 3: Interrogative-subject agreement
        who_result = self._check_who_agreement(text)
        if not who_result.is_valid:
            logger.debug("REJECT (%s): %r", who_result.reason, text[:80])
            return who_result

        # Check 4: Must be interrogative or imperative
        if self._is_purely_declarative(text):
            logger.debug("REJECT (declarative_not_instruction): %r", text[:80])
            return SanityResult(is_valid=False, reason="declarative_not_instruction")

        logger.debug("ACCEPT: %r", text[:80])
        return SanityResult(is_valid=True)

    def _check_generic_text(self, sample: dict) -> SanityResult:
        """Fallback check for samples without an instruction field."""
        text_keys = ["text", "content", "question"]
        for key in text_keys:
            val = sample.get(key)
            if isinstance(val, str) and val.strip():
                if len(val.strip()) < _MIN_INSTRUCTION_LENGTH:
                    return SanityResult(is_valid=False, reason="too_short")
                if self._has_excessive_repetition(val):
                    return SanityResult(is_valid=False, reason="garbled_repetition")
                return SanityResult(is_valid=True)
        return SanityResult(is_valid=True)  # No text to check; pass through

    # ------------------------------------------------------------------
    # Who-agreement check (spaCy + rule fallback)
    # ------------------------------------------------------------------

    def _check_who_agreement(self, text: str) -> SanityResult:
        """Ensure 'Who' questions target persons/orgs, not inanimate concepts."""
        stripped = text.strip()

        # Only applies to questions starting with "Who"
        if not re.match(r"^who\b", stripped, re.IGNORECASE):
            return SanityResult(is_valid=True)

        # Must end with a question mark (or be clearly interrogative)
        if not stripped.endswith("?"):
            return SanityResult(is_valid=False, reason="who_missing_question_mark")

        if _nlp is not None:
            return self._check_who_with_spacy(stripped)
        else:
            return self._check_who_with_rules(stripped)

    def _check_who_with_spacy(self, text: str) -> SanityResult:
        """Use spaCy NER to verify the subject of a 'Who' question."""
        doc = _nlp(text)

        entities = [ent for ent in doc.ents]

        # If spaCy finds entities, check if at least one is a person/org
        if entities:
            valid_entity_found = any(
                ent.label_ in _WHO_VALID_LABELS for ent in entities
            )
            if valid_entity_found:
                return SanityResult(is_valid=True)
            else:
                return SanityResult(
                    is_valid=False,
                    reason="who_targets_inanimate",
                )

        # No entities found — use heuristic: check if the subject after
        # "Who is/are" is likely a concept rather than a named entity.
        return self._check_who_subject_heuristic(text)

    def _check_who_with_rules(self, text: str) -> SanityResult:
        """Rule-based fallback when spaCy is not available."""
        return self._check_who_subject_heuristic(text)

    def _check_who_subject_heuristic(self, text: str) -> SanityResult:
        """
        Heuristic: after 'Who is/are/was/were', if the remaining words
        are all lowercase and contain known concept indicators, reject.
        """
        match = re.match(
            r"^who\s+(?:is|are|was|were)\s+(.+?)\??$",
            text,
            re.IGNORECASE,
        )
        if not match:
            return SanityResult(is_valid=True)

        subject = match.group(1).strip()

        # Known inanimate concept indicators
        inanimate_indicators = [
            "computing", "computer", "physics", "science", "technology",
            "engineering", "mathematics", "math", "chemistry", "biology",
            "programming", "software", "hardware", "internet", "data",
            "algorithm", "machine learning", "deep learning", "neural",
            "quantum", "gravity", "electricity", "energy", "force",
        ]

        subject_lower = subject.lower()
        for indicator in inanimate_indicators:
            if indicator in subject_lower:
                return SanityResult(
                    is_valid=False,
                    reason="who_targets_inanimate",
                )

        # If subject is entirely lowercase with no capitalized proper nouns,
        # it is probably a concept, not a person.
        words = subject.split()
        if len(words) >= 1 and all(w[0].islower() for w in words if w):
            return SanityResult(
                is_valid=False,
                reason="who_targets_inanimate",
            )

        return SanityResult(is_valid=True)

    # ------------------------------------------------------------------
    # Declarative detection
    # ------------------------------------------------------------------

    def _is_purely_declarative(self, text: str) -> bool:
        """
        Detect if text is a plain declarative statement rather than
        a question or imperative command.

        Heuristic: If it doesn't end with '?' and doesn't start with
        an imperative verb or interrogative word, it's declarative.
        """
        stripped = text.strip()

        # Questions are fine
        if stripped.endswith("?"):
            return False

        # Check for imperative starters (common instruction prefixes)
        imperative_starters = [
            "explain", "describe", "list", "define", "compare",
            "summarize", "analyze", "evaluate", "discuss", "identify",
            "outline", "provide", "give", "name", "state", "tell",
            "write", "create", "generate", "translate", "convert",
            "calculate", "compute", "find", "determine", "show",
            "prove", "classify", "categorize", "rank", "sort",
        ]

        first_word = stripped.split()[0].lower() if stripped.split() else ""
        if first_word in imperative_starters:
            return False

        # Check for interrogative words (might be missing '?')
        interrogative_starters = [
            "what", "how", "why", "when", "where", "which", "who",
            "whom", "whose", "can", "could", "would", "should",
            "do", "does", "did", "is", "are", "was", "were",
            "will", "shall", "have", "has", "had",
        ]
        if first_word in interrogative_starters:
            return False

        # Anything else is likely declarative
        return True

    # ------------------------------------------------------------------
    # Repetition check
    # ------------------------------------------------------------------

    def _has_excessive_repetition(self, text: str) -> bool:
        """Detect garbled LLM output with excessive word repetition."""
        words = text.lower().split()
        if len(words) < 4:
            return False

        from collections import Counter
        counts = Counter(words)
        most_common_count = counts.most_common(1)[0][1]
        ratio = most_common_count / len(words)
        return ratio > _MAX_WORD_REPEAT_RATIO

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_instruction_text(
        self, sample: dict, instruction_key: str
    ) -> str | None:
        """Extract instruction text from a sample dict."""
        for key in [instruction_key, "question", "instruction", "input"]:
            val = sample.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return None
