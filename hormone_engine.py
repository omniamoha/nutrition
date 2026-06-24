# ===============================
# Generic Hormone Engine
# ===============================

def get_range(config, age_group, gender):
    """
    Get reference range based on age & gender
    """

    ranges = config["ranges"]

    val = ranges.get(age_group, config.get("default_range", (0, 100)))

    if isinstance(val, dict):
        return val.get(gender, config.get("default_range", (0, 100)))

    return val


# ===============================
# Analysis
# ===============================
def analyze(value, config, age_group, gender):

    normal_min, normal_max = get_range(config, age_group, gender)

    if value < normal_min:
        status = "Low"
    elif value > normal_max:
        status = "High"
    else:
        status = "Normal"

    message = f"{status} ({normal_min}-{normal_max})"

    return status, message


# ===============================
# Scoring Engine
# ===============================
def apply_score(df, config, status):

    df = df.copy()
    score = 0

    for rule in config["scoring"][status]:
        col = rule["column"]
        weight = rule["weight"]

        score += df.get(col, 0) * weight

    df[config["score_name"]] = score

    return df


# ===============================
# Full Pipeline
# ===============================
def process_hormone(df, value, config, age_group, gender):

    status, message = analyze(value, config, age_group, gender)

    df = apply_score(df, config, status)

    df_sorted = df.sort_values(
        by=config["score_name"],
        ascending=False
    ).reset_index(drop=True)

    advice = config["advice"][status]

    return df_sorted, status, message, advice