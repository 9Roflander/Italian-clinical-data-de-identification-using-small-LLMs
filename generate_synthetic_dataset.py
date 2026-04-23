#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic Italian clinical notes using Anthropic Claude 3 Haiku.

Key features implemented:
- Model: claude-3-haiku-20240307
- Prompt Caching with a >=2048 token cached system block
- Exactly 20 permanent anchor records from gold_standard_80.json
- Dynamic sampling of 3 records from the remaining 60 on each iteration
- Offset calculation via original_text.find(entity['text'])
- Backup save every 10 generated records
"""

import argparse
import json
import logging
import os
import random
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from anthropic import Anthropic


MODEL_NAME = "claude-3-haiku-20240307"
MIN_STATIC_TOKENS = 2048
ANCHOR_COUNT = 20
DYNAMIC_SAMPLE_COUNT = 3
EXPECTED_GOLD_COUNT = 80


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("synthetic_generator")


SYSTEM_INSTRUCTIONS_IT = """Sei un esperto di Informatica Medica e GDPR. Genera note cliniche sintetiche e realistiche in italiano.

REGOLE TASSATIVE PER original_text:
- Usa NOMI PROPRI REALI (es. 'Giuseppe Verdi', 'Dott.ssa Maria Bianchi').
- Istruzioni: Quando inserisci dati identificativi, usa esattamente questi segnaposto nel campo original_text:
  [PLACEHOLDER_CF] per il Codice Fiscale.
  [PLACEHOLDER_TEL] per il Numero di Telefono.
  [PLACEHOLDER_ID] per altri ID (Cartella Clinica, etc.).
- Esempio: 'Il paziente [PLACEHOLDER_CF] è stato contattato al [PLACEHOLDER_TEL].'
- NON USARE MAI segnaposto come [NOME] o 'NOME' nel campo original_text per i nomi, usa solo quelli sopra per gli ID.
- Il testo deve sembrare un vero referto ospedaliero.

REGOLE PER redacted_text:
- Deve essere identico all'originale, ma con i dati identificativi (inclusi i placeholder) e personali sostituiti dai tag: [NOME], [ETÀ], [DATA], [LUOGO], [ID].

REGOLE PER entities:
- Elenca ogni dato inserito (compresi i segnaposto esatti, es. "[PLACEHOLDER_CF]") nel testo originale con il relativo tag.
"""


def approx_token_count(text: str) -> int:
    """Rough token estimate used to guarantee a long cacheable system block."""
    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def ensure_minimum_tokens(text: str, min_tokens: int) -> str:
    """Pad the static prompt until it likely exceeds min token threshold."""
    if approx_token_count(text) >= min_tokens:
        return text

    filler = (
        "\nLinea guida aggiuntiva: mantieni coerenza interna tra storia clinica, "
        "anamnesi, esame obiettivo, terapia e follow-up; usa sempre dati fittizi "
        "coerenti e non riutilizzare dettagli letterali degli esempi ancora."
    )
    padded = text
    while approx_token_count(padded) < min_tokens:
        padded += filler
    return padded


def load_gold_records(input_path: Path) -> List[Dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(
            f"File non trovato: {input_path}. Metti gold_standard_80.json nella stessa cartella dello script."
        )

    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("gold_standard_80.json deve contenere una lista di record.")
    if len(data) != EXPECTED_GOLD_COUNT:
        raise ValueError(
            f"Attesi {EXPECTED_GOLD_COUNT} record in gold_standard_80.json, trovati: {len(data)}"
        )

    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"Record {idx} non valido: ogni elemento deve essere un dizionario.")
        if "text" not in rec:
            raise ValueError(f"Record {idx} privo del campo obbligatorio 'text'.")
        if "entities" not in rec:
            raise ValueError(f"Record {idx} privo del campo obbligatorio 'entities'.")

    return data


def select_anchor_and_pool(
    records: Sequence[Dict[str, Any]],
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)
    indices = list(range(len(records)))
    anchor_indices = set(rng.sample(indices, ANCHOR_COUNT))

    anchors = [records[i] for i in indices if i in anchor_indices]
    dynamic_pool = [records[i] for i in indices if i not in anchor_indices]

    if len(anchors) != ANCHOR_COUNT:
        raise RuntimeError("Selezione anchor non valida: numero diverso da 20.")
    if len(dynamic_pool) != EXPECTED_GOLD_COUNT - ANCHOR_COUNT:
        raise RuntimeError("Pool dinamico non valido: attesi 60 record.")

    return anchors, dynamic_pool


def build_cached_system_prompt(anchor_records: Sequence[Dict[str, Any]]) -> str:
    anchor_block = json.dumps(list(anchor_records), ensure_ascii=False, indent=2)

    static_block = f"""{SYSTEM_INSTRUCTIONS_IT}

Contesto operativo permanente (blocco statico cacheabile):
- Le note devono sembrare estratti realistici da cartelle cliniche italiane.
- Mantieni stile professionale: reparto, anamnesi patologica remota, esame obiettivo, diagnosi, terapia.
- Le entità devono essere coerenti con il testo e con la redazione.
- Non inventare chiavi JSON aggiuntive: solo original_text, redacted_text, entities.
- In entities usa sempre testo originale non redatto.
- Evita nomi di ospedali reali famosi; preferisci strutture e località plausibili ma fittizie.
- Inserisci almeno 4-8 entità sensibili complessive quando clinicamente plausibile.

Record ancora permanenti (esattamente 20) da usare solo come riferimento di stile, struttura e granularità:
{anchor_block}

Regola critica finale:
- Devi rispondere con un singolo oggetto JSON valido, senza markdown e senza testo extra.
"""

    return ensure_minimum_tokens(static_block, MIN_STATIC_TOKENS)


def build_dynamic_user_prompt(
    iteration: int,
    dynamic_examples: Sequence[Dict[str, Any]],
) -> str:
    examples_json = json.dumps(list(dynamic_examples), ensure_ascii=False, indent=2)

    return f"""Genera 1 nuova nota clinica sintetica originale (record #{iteration}).

Vincoli operativi:
1) Non copiare frasi o dettagli letterali dagli esempi.
2) Mantieni registro medico italiano professionale.
3) Coerenza perfetta tra original_text, entities e redacted_text.
4) Ogni elemento di entities deve comparire testualmente in original_text.
5) redacted_text deve essere identico a original_text salvo sostituzioni con tag [NOME], [ETÀ], [DATA], [LUOGO], [ID].
6) Restituisci solo JSON valido.

Ispirazione dinamica (3 record estratti casualmente):
{examples_json}
"""


def extract_text_from_response(response: Any) -> str:
    chunks: List[str] = []
    for block in getattr(response, "content", []):
        if getattr(block, "type", None) == "text" and hasattr(block, "text"):
            chunks.append(block.text)
    return "\n".join(chunks).strip()


def parse_json_object(raw_text: str) -> Dict[str, Any]:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Risposta del modello non contiene un oggetto JSON valido.")

    obj = json.loads(cleaned[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Il JSON generato deve essere un oggetto.")
    return obj


def normalize_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def canonical_type(raw_type: str) -> str:
    normalized = unicodedata.normalize("NFKD", raw_type.upper())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))

    if any(key in normalized for key in ["NOME", "COGNOME", "PAZIENT", "MEDIC"]):
        return "NOME"
    if any(key in normalized for key in ["ETA", "ANNI", "AGE"]):
        return "ETÀ"
    if any(key in normalized for key in ["DATA", "GIORNO", "MESE", "ANNO"]):
        return "DATA"
    if any(
        key in normalized
        for key in ["LUOGO", "INDIRIZ", "CITTA", "COMUNE", "PROVINCIA", "OSPEDALE"]
    ):
        return "LUOGO"
    if any(
        key in normalized
        for key in ["ID", "CODICE", "CF", "TESSERA", "CARTELLA", "MATRICOLA", "NUMERO", "PLACEHOLDER_CF", "PLACEHOLDER_TEL", "PLACEHOLDER_ID"]
    ):
        return "ID"
    return "ID"


def type_to_tag(entity_type: str) -> str:
    mapping = {
        "NOME": "[NOME]",
        "ETÀ": "[ETÀ]",
        "DATA": "[DATA]",
        "LUOGO": "[LUOGO]",
        "ID": "[ID]",
    }
    return mapping.get(entity_type, "[ID]")


def find_non_overlapping_span(
    text: str,
    needle: str,
    occupied_spans: List[Tuple[int, int]],
) -> Tuple[int, int]:
    # Requirement-specific strategy: span detection based on str.find.
    start = text.find(needle)
    while start != -1:
        end = start + len(needle)
        overlaps = any(not (end <= s or start >= e) for s, e in occupied_spans)
        if not overlaps:
            occupied_spans.append((start, end))
            return start, end
        start = text.find(needle, start + 1)
    raise ValueError(f"Entità non trovata in original_text: '{needle}'")


def enrich_entities_with_offsets(
    original_text: str,
    raw_entities: Any,
) -> List[Dict[str, Any]]:
    if not isinstance(raw_entities, list):
        raise ValueError("Il campo entities deve essere una lista.")

    occupied: List[Tuple[int, int]] = []
    enriched: List[Dict[str, Any]] = []
    for item in raw_entities:
        if not isinstance(item, dict):
            continue
        text = normalize_text(item.get("text"))
        entity_type = canonical_type(normalize_text(item.get("type", "ID")))
        if not text:
            continue

        start, end = find_non_overlapping_span(original_text, text, occupied)
        enriched.append(
            {
                "type": entity_type,
                "text": text,
                "start": start,
                "end": end,
            }
        )

    enriched.sort(key=lambda x: (x["start"], x["end"]))
    return enriched


def rebuild_redacted_text(original_text: str, entities: Sequence[Dict[str, Any]]) -> str:
    if not entities:
        return original_text

    parts: List[str] = []
    cursor = 0
    for entity in entities:
        start = int(entity["start"])
        end = int(entity["end"])
        if start < cursor:
            continue
        parts.append(original_text[cursor:start])
        parts.append(type_to_tag(str(entity["type"])))
        cursor = end
    parts.append(original_text[cursor:])
    return "".join(parts)


def validate_and_normalize_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    original_text = normalize_text(obj.get("original_text"))
    if not original_text:
        raise ValueError("original_text mancante o vuoto.")
    if re.search(r"\[(?:NOME|ETÀ|DATA|LUOGO|ID)\]", original_text):
        raise ValueError("original_text contiene tag redazionali: risposta non valida.")

    entities = enrich_entities_with_offsets(original_text, obj.get("entities", []))
    if not entities:
        raise ValueError("entities vuoto o non allineabile con original_text.")

    redacted_text = rebuild_redacted_text(original_text, entities)

    return {
        "original_text": original_text,
        "redacted_text": redacted_text,
        "entities": entities,
    }


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def create_with_prompt_caching(client: Anthropic, **kwargs: Any) -> Any:
    # Prefer dedicated prompt-caching API when available, then beta.messages.create.
    beta_client = getattr(client, "beta", None)
    if beta_client and hasattr(beta_client, "prompt_caching"):
        return beta_client.prompt_caching.messages.create(**kwargs)

    if beta_client and hasattr(beta_client, "messages"):
        beta_kwargs = dict(kwargs)
        beta_kwargs["betas"] = ["prompt-caching-2024-07-31"]
        return beta_client.messages.create(**beta_kwargs)

    kwargs = dict(kwargs)
    headers = dict(kwargs.get("extra_headers") or {})
    headers["anthropic-beta"] = "prompt-caching-2024-07-31"
    kwargs["extra_headers"] = headers
    return client.messages.create(**kwargs)


def generate_dataset(
    input_path: Path,
    output_path: Path,
    n_records: int,
    seed: int,
    max_retries: int,
) -> None:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError("Variabile ambiente ANTHROPIC_API_KEY non impostata.")

    gold_records = load_gold_records(input_path)
    anchor_records, dynamic_pool = select_anchor_and_pool(gold_records, seed)
    cached_system_prompt = build_cached_system_prompt(anchor_records)

    approx_tokens = approx_token_count(cached_system_prompt)
    logger.info("Static system prompt costruito (stima token: %s).", approx_tokens)
    if approx_tokens < MIN_STATIC_TOKENS:
        raise RuntimeError("Static system prompt troppo corto: caching non garantito.")

    client = Anthropic(api_key=api_key)
    rng = random.Random(seed + 1)
    synthetic_records: List[Dict[str, Any]] = []

    backup_path = output_path.with_name(f"{output_path.stem}_backup.json")

    for i in range(1, n_records + 1):
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                dynamic_examples = rng.sample(dynamic_pool, DYNAMIC_SAMPLE_COUNT)
                user_prompt = build_dynamic_user_prompt(i, dynamic_examples)
                if i % 2 == 0:
                    user_prompt += " Inserisci obbligatoriamente [PLACEHOLDER_CF] e [PLACEHOLDER_TEL] nel testo."

                response = create_with_prompt_caching(
                    client,
                    model=MODEL_NAME,
                    max_tokens=1200,
                    temperature=0.4,
                    system=[
                        {
                            "type": "text",
                            "text": cached_system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_prompt}],
                        }
                    ],
                )

                response_text = extract_text_from_response(response)
                parsed = parse_json_object(response_text)
                normalized = validate_and_normalize_record(parsed)
                synthetic_records.append(normalized)
                success = True
                logger.info("Record %s/%s generato con successo.", i, n_records)
                break
            except Exception as exc:
                logger.warning(
                    "Tentativo %s/%s fallito per record %s: %s",
                    attempt,
                    max_retries,
                    i,
                    exc,
                )
                if attempt == max_retries:
                    raise RuntimeError(f"Impossibile generare record {i} dopo {max_retries} tentativi.") from exc
            finally:
                # Requirement: throttle requests to reduce rate-limit risk.
                time.sleep(1)

        if not success:
            raise RuntimeError(f"Record {i} non generato.")

        if i % 10 == 0:
            save_json(backup_path, synthetic_records)
            save_json(output_path, synthetic_records)
            logger.info("Backup salvato a %s (record: %s).", backup_path, i)

    save_json(output_path, synthetic_records)
    logger.info("Dataset finale salvato in %s (totale record: %s).", output_path, len(synthetic_records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Italian clinical de-identification dataset via Anthropic Claude 3 Haiku."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("gold_standard_80.json"),
        help="Path al file gold_standard_80.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("synthetic_clinical_100.json"),
        help="Path al file JSON di output",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=100,
        help="Numero di record sintetici da generare (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per campionamento riproducibile",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Numero massimo di retry per record",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        input_path=args.input,
        output_path=args.output,
        n_records=args.n_records,
        seed=args.seed,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
