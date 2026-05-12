import json
import csv
import pandas as pd
import ollama
from tqdm import tqdm
import argparse
from pydantic import BaseModel
from typing import Literal
import os
#import vllm
import ollama
from openai import OpenAI


#choose GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,5,6"

class Annotation(BaseModel):
    text: str
    type: str

class AnnotationDeidentified(BaseModel):
    text: str
    type: str
    counted_as: Literal["TP", "FN", "FP"]  # This enforces only these three values are allowed

class EvaluationResult(BaseModel):
    report_id: str
    annotations_gold: list[Annotation]
    annotations_deidentified: list[AnnotationDeidentified]

evaluation_prompt = """
Ti fornirò:
 • Il testo originale di un referto medico (testo_originale)
 • La sua versione anonimizzata (testo_anonimizzato)
 • Una lista di entità sensibili annotate manualmente (entità_gold)

Le possibili categorie sono:
- NOME
- ETÀ
- LUOGO/INDIRIZZO
- DATA

Il tuo compito è confrontare le entità del gold standard con quelle effettivamente anonimizzate nel testo.

Per ciascuna entità del gold, verifica:
 • Se è stata correttamente anonimizzata, il testo dell'entità gold è stato sostituito con il tag corrispondente alla categoria: mettila in annotations_deidentified con counted_as: "TP"
 ESEMPIO:
 • Entità gold: "Mario Rossi"
 • Entità deidentified: "[NOME]"
 • Output: "Mario Rossi", "NOME", "TP"
 
 • Se non è stata correttamente anonimizzata, il testo dell'entità gold è rimasto invariato: mettila in annotations_deidentified con counted_as: "FN"
 ESEMPIO:
 • Entità gold: "Mario Rossi"
 • Entità deidentified: "Mario Rossi"
 • Output: "Mario Rossi", "NOME", "FN"
 
È possibile che compaiano entità anonimizzate che non sono presenti nel gold standard. Questo vuol dire che è stato anonimizzato un testo che non conteneva entità sensibili. In questo caso, mettila in annotations_deidentified con counted_as: "FP"
ESEMPIO:
 • Entità deidentified: "[NOME]"
 • Output: "[NOME]", "NOME", "FP"

IMPORTANTE: Ogni elemento in annotations_deidentified DEVE avere esattamente questi campi:
- text: il testo dell'entità
- type: il tipo dell'entità
- counted_as: deve essere esattamente "TP", "FN", o "FP"

NOTA: Ogni entità gold deve in qualche modo essere presente nel testo anonimizzato e sarà contata come "TP" se è stata anonimizzata correttamente, "FN" se non è stata anonimizzata. Questo significa che la cardinalità di annotations_deidentified deve essere maggiore o uguale alla cardinalità di annotations_gold.

ATTENZIONE: 
- Ogni output deve essere un JSON valido, verrà poi processato con json.loads().
- Non aggiungere altro testo oltre al JSON, altrimenti verrà considerato un errore.
- Assicurati di mettere tra virgolette TUTTI i valori di testo, inclusi i tag come [NOME], [ETÀ], etc.
- Non usare virgole al posto dei due punti nelle coppie chiave-valore.

ESEMPI:
--NOME
Esempio di output:
{"report_id": "1", "annotations_gold": [{"text": "Mario Rossi", "type": "NOME"}, {"text": "Giovanni Di Lorenzo", "type": "NOME"}], "annotations_deidentified": [{"text": "Mario Rossi", "type": "NOME", "counted_as": "FN"}, {"text": "[NOME]", "type": "NOME", "counted_as": "TP"}, {"text": "[NOME]", "type": "NOME", "counted_as": "FP"}]}

--ETÀ
Esempio di output:
{"report_id": "1", "annotations_gold": [{"text": "25", "type": "ETÀ"}, {"text": "30", "type": "ETÀ"}], "annotations_deidentified": [{"text": "25", "type": "ETÀ", "counted_as": "FN"}, {"text": "[ETÀ]", "type": "ETÀ", "counted_as": "TP"}, {"text": "[ETÀ]", "type": "ETÀ", "counted_as": "FP"}]}

--LUOGO/INDIRIZZO
Esempio di output:
{"report_id": "1", "annotations_gold": [{"text": "Pakistan", "type": "LUOGO/INDIRIZZO"}, {"text": "Bologna", "type": "LUOGO/INDIRIZZO"}], "annotations_deidentified": [{"text": "[LUOGO/INDIRIZZO]", "type": "LUOGO/INDIRIZZO", "counted_as": "TP"}, {"text": "Bologna", "type": "LUOGO/INDIRIZZO", "counted_as": "FN"}, {"text": "[LUOGO/INDIRIZZO]", "type": "LUOGO/INDIRIZZO", "counted_as": "FP"}]}

--DATA
Esempio di output:
{"report_id": "1", "annotations_gold": [{"text": "2021-01-01", "type": "DATA"}, {"text": "4 Maggio", "type": "DATA"}], "annotations_deidentified": [{"text": "2021-01-01", "type": "DATA", "counted_as": "FN"}, {"text": "[DATA]", "type": "DATA", "counted_as": "TP"}, {"text": "[DATA]", "type": "DATA", "counted_as": "FP"}]}
"""

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="gemma3:27b")
parser.add_argument("--test_length", type=int, default=None)
parser.add_argument("--category", type=str, default="NOME")
parser.add_argument("--deidentified_data_path", type=str)
parser.add_argument("--backend", type=str, default="ollama")
parser.add_argument("--temperature", type=float, default=0.7)
args = parser.parse_args()

annotated_data_path = "Annotated clinical notes - samples.csv"
sensitive_information_categories = ["NOME","ETÀ","LUOGO/INDIRIZZO","DATA"]

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="-",
)



annotated_data = []
with open(annotated_data_path, 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip header row
    for row in csv_reader:
        annotated_data.append({
            "text": row[0],
            "annotations_redacted": row[1]
        })

annotations = []
for row in annotated_data:
    annotations.append(json.loads(row["annotations_redacted"]))

deidentified_data = []
input_data = []
with open(args.deidentified_data_path, 'r') as f:
    for line in f:
        input_data.append(json.loads(line)["input"])
        deidentified_data.append(json.loads(line)["output"])


def evaluate_on_category(input_data,deidentified_data,annotations,category,test_length):
    results = []
    if test_length is not None:
        input_data = input_data[:test_length]
        deidentified_data = deidentified_data[:test_length]
        annotations = annotations[:test_length]
    for (input, deidentified, annotation) in tqdm(zip(input_data,deidentified_data,annotations)):
        #filter annotations by category
        annotations_category = [a for a in annotation if a["type"] == category]
        
        #prompt the model to evaluate the deidentified data
        prompt = evaluation_prompt + f"TESTO ORIGINALE: {input}\nTESTO ANONIMIZZATO: {deidentified}\nENTITÀ GOLD: {annotations_category}"
        if args.backend == "ollama":
            
            ollama.Client(host='http://127.0.0.1:11435')  # Use the new port

            response = ollama.generate(model=args.model, prompt=prompt, format=EvaluationResult.model_json_schema())
            response = json.loads(response["response"])
            #breakpoint()
            #filter the response to exclude entities that are from different categories
            response["annotations_deidentified"] = [annotation for annotation in response["annotations_deidentified"] if annotation["type"] == category]
            response["annotations_gold"] = [ann for ann in annotation if ann["type"] == category]
            results.append(response)

        elif args.backend == "vllm":
            completion = client.chat.completions.create(
                model=args.model,
                messages=[
                    {"role": "user", "content": evaluation_prompt}
                ],
                extra_body={"guided_json": {"schema": EvaluationResult.model_json_schema()}},
                temperature=args.temperature,
                max_tokens=4096
            )
            breakpoint()
            results.append(completion.choices[0].message.content)
        
    
    # Sanitize category name for filename
    safe_category = category.replace("/", "_").replace("\\", "_")
    print(args.deidentified_data_path.split('/')[-1].split('_')[0])
    #breakpoint()
    with open(f"outputs/{args.deidentified_data_path.split('/')[-1][:-5]}_evaluatedBy{args.model}_{safe_category}.jsonl", "w") as f:
        for result in results:
            try:
                f.write(json.dumps(result) + "\n")
            except Exception as e:
                print(e)
                breakpoint()

evaluate_on_category(input_data,deidentified_data,annotations,args.category,args.test_length)