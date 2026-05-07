#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate synthetic Italian clinical notes via a two-step pipeline:
  1) Gemini 2.5 Flash produces plain-text notes with PII placeholders.
  2) Faker (it_IT) injects realistic PII and computes entity offsets locally.
"""

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from faker import Faker
from google import genai
from google.genai import types as genai_types


MODEL_NAME = "gemini-2.5-flash"
TEMPERATURE = 0.8
MIN_ENTITIES_PER_RECORD = 8

# Placeholders the LLM must emit verbatim. Order matters for prompt readability only.
PLACEHOLDERS: Tuple[str, ...] = (
    "[PAZIENTE_NOME]",
    "[PAZIENTE_COGNOME]",
    "[DATA_NASCITA]",
    "[LUOGO_RESIDENZA]",
    "[CF]",
    "[TELEFONO]",
    "[MEDICO_NOME]",
    "[MEDICO_COGNOME]",
    "[PARENTE_NOME]",
    "[DATA_VISITA]",
)

PLACEHOLDER_TO_TYPE: Dict[str, str] = {
    "[PAZIENTE_NOME]": "NOME",
    "[PAZIENTE_COGNOME]": "NOME",
    "[DATA_NASCITA]": "DATA",
    "[LUOGO_RESIDENZA]": "LUOGO",
    "[CF]": "ID",
    "[TELEFONO]": "ID",
    "[MEDICO_NOME]": "NOME",
    "[MEDICO_COGNOME]": "NOME",
    "[PARENTE_NOME]": "NOME",
    "[DATA_VISITA]": "DATA",
}

TYPE_TO_TAG: Dict[str, str] = {
    "NOME": "[NOME]",
    "ETÀ": "[ETÀ]",
    "DATA": "[DATA]",
    "LUOGO": "[LUOGO]",
    "ID": "[ID]",
}

PLACEHOLDER_PATTERN = re.compile(
    r"\[(?:PAZIENTE_NOME|PAZIENTE_COGNOME|DATA_NASCITA|LUOGO_RESIDENZA|CF|TELEFONO|MEDICO_NOME|MEDICO_COGNOME|PARENTE_NOME|DATA_VISITA)\]"
)

# Light topic variation to fight LLM repetition without leaking anchor records.
SPECIALTIES_IT = (
    "Cardiologia", "Pneumologia", "Gastroenterologia", "Neurologia",
    "Ortopedia", "Endocrinologia", "Oncologia", "Nefrologia",
    "Pronto Soccorso", "Medicina Interna", "Geriatria", "Reumatologia",
    "Ematologia", "Urologia", "Chirurgia Generale", "Ginecologia",
)

NOTE_TYPES_IT = (
    "verbale di pronto soccorso",
    "lettera di dimissione",
    "diario clinico di reparto",
    "consulenza specialistica",
    "referto ambulatoriale",
    "anamnesi e visita di accettazione",
)


SYSTEM_PROMPT = """Sei un medico italiano esperto che redige documentazione clinica realistica.

REGOLE TASSATIVE PER L'OUTPUT:
- Restituisci SOLO testo semplice. Niente JSON, niente markdown.
- Usa esattamente questi segnaposto letterali: [PAZIENTE_NOME], [PAZIENTE_COGNOME], [DATA_NASCITA], [LUOGO_RESIDENZA], [CF], [TELEFONO], [MEDICO_NOME], [MEDICO_COGNOME], [PARENTE_NOME], [DATA_VISITA].
- DIVIETO DI RAGGRUPPAMENTO: NON inserire tutti i segnaposto nella prima frase o nello stesso blocco.

DISTRIBUZIONE NATURALE E CAOTICA (MOLTO IMPORTANTE):
Spargi i dati in tutto il testo in modo imprevedibile.
- [PAZIENTE_NOME], [PAZIENTE_COGNOME] e [DATA_NASCITA]: inseriscili nell'intestazione o nell'anamnesi.
- Terze Parti (Parenti): Fai in modo che il paziente sia accompagnato o supportato da un familiare. Esempio: "Il paziente è giunto in PS accompagnato dal figlio [PARENTE_NOME]" oppure "La moglie [PARENTE_NOME] riferisce che...".
- Terze Parti (Medici): Inserisci la firma o la menzione del medico curante. Esempio: "Si consiglia follow-up con il Dott. [MEDICO_NOME] [MEDICO_COGNOME]."
- Date Mediche: Usa [DATA_VISITA] per indicare la data di ammissione, di un intervento o della dimissione (es. "Ricoverato in data [DATA_VISITA]").
- [TELEFONO] e [CF]: NON metterli sempre in fondo al documento in un elenco ordinato. Nascondili nel testo.

- Varia enormemente lo stile della nota.
- Includi tutti i segnaposto almeno una volta.
"""


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("synthetic_generator_v2")


def build_user_prompt(iteration: int, rng: random.Random) -> str:
    specialty = rng.choice(SPECIALTIES_IT)
    note_type = rng.choice(NOTE_TYPES_IT)
    return (
        f"Genera UNA nuova nota clinica sintetica originale (record #{iteration}).\n"
        f"Reparto/specialità: {specialty}.\n"
        f"Tipo di documento: {note_type}.\n"
        "La nota deve essere clinicamente plausibile, con dettagli specifici "
        "(parametri vitali, esami, terapia) e includere TUTTI i segnaposto richiesti."
    )


def generate_fake_values(faker: Faker) -> Dict[str, str]:
    """One realistic value per placeholder, reused for every occurrence in the note."""
    dob = faker.date_of_birth(minimum_age=18, maximum_age=95)
    visit_date = faker.date_between(start_date="-5y", end_date="today")
    return {
        "[PAZIENTE_NOME]": faker.first_name(),
        "[PAZIENTE_COGNOME]": faker.last_name(),
        "[DATA_NASCITA]": dob.strftime("%d/%m/%Y"),
        "[LUOGO_RESIDENZA]": faker.city(),
        "[CF]": faker.ssn(),
        "[TELEFONO]": faker.phone_number(),
        "[MEDICO_NOME]": faker.first_name(),
        "[MEDICO_COGNOME]": faker.last_name(),
        "[PARENTE_NOME]": faker.first_name(),
        "[DATA_VISITA]": visit_date.strftime("%d/%m/%Y"),
    }


def inject_pii(text: str, fake_values: Dict[str, str]) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace placeholders with Faker values and record exact offsets in the new text."""
    out_parts: List[str] = []
    entities: List[Dict[str, Any]] = []
    cursor = 0
    out_len = 0

    for match in PLACEHOLDER_PATTERN.finditer(text):
        placeholder = match.group(0)
        before = text[cursor:match.start()]
        out_parts.append(before)
        out_len += len(before)

        value = fake_values[placeholder]
        start = out_len
        end = out_len + len(value)
        out_parts.append(value)
        out_len = end
        cursor = match.end()

        entities.append({
            "type": PLACEHOLDER_TO_TYPE[placeholder],
            "text": value,
            "start": start,
            "end": end,
        })

    out_parts.append(text[cursor:])
    return "".join(out_parts), entities


def build_redacted_text(original_text: str, entities: List[Dict[str, Any]]) -> str:
    if not entities:
        return original_text
    ordered = sorted(entities, key=lambda e: e["start"])
    parts: List[str] = []
    cursor = 0
    for ent in ordered:
        parts.append(original_text[cursor:ent["start"]])
        parts.append(TYPE_TO_TAG[ent["type"]])
        cursor = ent["end"]
    parts.append(original_text[cursor:])
    return "".join(parts)


def clean_llm_output(raw: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:[a-zA-Z]+)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def validate_placeholders(text: str) -> None:
    found = PLACEHOLDER_PATTERN.findall(text)
    if not found:
        raise ValueError("Nessun segnaposto presente nel testo generato.")
    if len(found) < MIN_ENTITIES_PER_RECORD:
        raise ValueError(
            f"Trovati solo {len(found)} segnaposto, attesi almeno {MIN_ENTITIES_PER_RECORD}."
        )


def call_gemini(client: genai.Client, user_prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=genai_types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            http_options=genai_types.HttpOptions(timeout=60_000),
        ),
    )
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Risposta Gemini vuota.")
    return clean_llm_output(text)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_dataset(
    output_path: Path,
    n_records: int,
    seed: int,
    max_retries: int,
    resume_from: int = 0,
) -> None:
    api_key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise EnvironmentError(
            "Variabile d'ambiente GEMINI_API_KEY (o GOOGLE_API_KEY) non impostata."
        )

    client = genai.Client(api_key=api_key)
    faker = Faker("it_IT")
    Faker.seed(seed)
    rng = random.Random(seed)

    backup_path = output_path.with_name(f"{output_path.stem}_backup.json")

    # Load existing records when resuming.
    synthetic_records: List[Dict[str, Any]] = []
    if resume_from > 0 and output_path.exists():
        synthetic_records = json.loads(output_path.read_text(encoding="utf-8"))
        logger.info("Ripreso da record %s: caricati %s record esistenti.", resume_from, len(synthetic_records))
    elif resume_from > 0 and backup_path.exists():
        synthetic_records = json.loads(backup_path.read_text(encoding="utf-8"))
        logger.info("Ripreso da backup: caricati %s record esistenti.", len(synthetic_records))

    # Advance RNG state to stay consistent with original sequence.
    for _ in range(resume_from):
        rng.choice(SPECIALTIES_IT)
        rng.choice(NOTE_TYPES_IT)

    for i in range(resume_from + 1, n_records + 1):
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                user_prompt = build_user_prompt(i, rng)
                template_text = call_gemini(client, user_prompt)
                validate_placeholders(template_text)

                fake_values = generate_fake_values(faker)
                original_text, entities = inject_pii(template_text, fake_values)
                if len(entities) < MIN_ENTITIES_PER_RECORD:
                    raise ValueError("Numero di entità iniettate inferiore al minimo.")

                redacted_text = build_redacted_text(original_text, entities)

                synthetic_records.append({
                    "original_text": original_text,
                    "redacted_text": redacted_text,
                    "entities": entities,
                })
                logger.info("Record %s/%s generato (entità: %s).", i, n_records, len(entities))
                break
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Tentativo %s/%s fallito per record %s: %s",
                    attempt, max_retries, i, exc,
                )
                time.sleep(1)
        else:
            raise RuntimeError(
                f"Impossibile generare record {i} dopo {max_retries} tentativi."
            ) from last_error

        if i % 10 == 0:
            save_json(backup_path, synthetic_records)
            save_json(output_path, synthetic_records)
            logger.info("Backup salvato a %s (record: %s).", backup_path, i)

    save_json(output_path, synthetic_records)
    logger.info("Dataset finale salvato in %s (totale: %s).", output_path, len(synthetic_records))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic Italian clinical notes with Gemini 2.5 Flash + Faker."
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
        help="Numero di record sintetici da generare",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per riproducibilità (Faker + scelta variazioni)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Numero massimo di retry per record",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Riprendi la generazione a partire da questo numero di record già completati",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(
        output_path=args.output,
        n_records=args.n_records,
        seed=args.seed,
        max_retries=args.max_retries,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
