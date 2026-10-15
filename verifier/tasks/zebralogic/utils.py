import json
import re


def zebralogic_doc_to_text(doc):
    puzzle = doc["puzzle"].strip()
    gold_solution = doc["solution"]
    columns = gold_solution["header"][1:]
    num_houses = len(gold_solution["rows"])

    house_keys = [f"House {i}" for i in range(1, num_houses + 1)]

    solution_lines = []
    for i, hk in enumerate(house_keys):
        inner = ", ".join([f'"{col}": "___"' for col in columns])
        comma = "," if i < len(house_keys) - 1 else ""
        solution_lines.append(f'    "{hk}": {{{inner}}}{comma}')
    solution_block = "\n".join(solution_lines)

    json_template = (
        "{\n"
        '  "reasoning": "___",\n'
        '  "solution": {\n'
        f"{solution_block}\n"
        "  }\n"
        "}"
    )

    example_puzzle = """There are 2 houses, numbered 1 to 2 from left to right, where each person lives in a different house. Each person has a unique name, a unique occupation, and a unique favorite animal.
1. The teacher is somewhere to the left of Arnold.
2. The person who loves cats is the doctor.
3. Eric is the teacher.
4. Arnold is the person who loves dogs."""

    example_answer = """{
  "reasoning": "From clue 3, Eric is the teacher. From clue 1, the teacher is left of Arnold, so Eric must be in House 1 and Arnold in House 2. From clue 4, Arnold loves dogs, so Eric must love cats. From clue 2, the person who loves cats is the doctor, so Eric is the doctor. Therefore Arnold has the remaining occupation.",
  "solution": {
    "House 1": {"Name": "Eric", "Occupation": "doctor", "Animal": "cats"},
    "House 2": {"Name": "Arnold", "Occupation": "teacher", "Animal": "dogs"}
  }
}"""

    return f"""# Example Puzzle
{example_puzzle}

# Answer to the Example Puzzle
{example_answer}

# Puzzle to Solve
{puzzle}

Now please solve the above puzzle. Present your reasoning and solution in the following json format:
{json_template}

Keep the reasoning concise and ensure the JSON is complete and valid.
"""


def extract_best_json_object(text: str):
    text = text.strip()
    candidates = []

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.S | re.I)
    candidates.extend(fenced)
    candidates.append(text)

    best = None
    best_score = -1

    for cand in candidates:
        starts = [i for i, ch in enumerate(cand) if ch == "{"]

        for start in starts:
            depth = 0
            end = None
            in_string = False
            escape = False

            for i in range(start, len(cand)):
                ch = cand[i]

                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break

            if end is None:
                continue

            chunk = cand[start:end]
            try:
                obj = json.loads(chunk)
            except Exception:
                continue

            score = 0
            if isinstance(obj, dict):
                if "solution" in obj:
                    score += 1_000_000
                if "reasoning" in obj:
                    score += 1_000
                score += len(chunk)

            if score > best_score:
                best = obj
                best_score = score

    return best


def zebralogic_process_results(doc, results):
    pred_text = results[0]
    parsed = extract_best_json_object(pred_text)

    gold = doc["solution"]
    columns = gold["header"]
    num_houses = len(gold["rows"])

    solution_table = {}
    total_cells = 0
    for i in range(num_houses):
        house_name = f"House {i+1}"
        solution_table[house_name] = {
            columns[j]: gold["rows"][i][j]
            for j in range(1, len(columns))
        }
        total_cells += len(columns) - 1

    if not isinstance(parsed, dict) or "solution" not in parsed or parsed["solution"] is None:
        return {"acc": 0.0, "cell_acc": 0.0, "noans_rate": 1.0}

    prediction_table = parsed["solution"]
    if not isinstance(prediction_table, dict):
        return {"acc": 0.0, "cell_acc": 0.0, "noans_rate": 1.0}

    correct_cells = 0
    for house in solution_table:
        for column in solution_table[house]:
            if house in prediction_table and column in prediction_table[house]:
                pred_val = prediction_table[house][column]

                if isinstance(pred_val, list):
                    pred_val = pred_val[0] if pred_val else None

                if pred_val is None or pred_val == "":
                    continue

                truth = str(solution_table[house][column]).lower().strip()
                pred = str(pred_val).lower().strip()

                if truth == pred:
                    correct_cells += 1

    acc = 1.0 if correct_cells == total_cells else 0.0
    cell_acc = correct_cells / total_cells if total_cells else 0.0

    return {"acc": acc, "cell_acc": cell_acc, "noans_rate": 0.0}