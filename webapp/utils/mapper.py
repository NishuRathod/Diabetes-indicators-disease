def map_inputs(form: dict) -> list[float]:
    """Convert form data into numeric input list."""
    mapping = {
        "Sex": {"Male": 1, "Female": 0},
        "GenHlth": {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5},
        "Age": {
            "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
            "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
            "70-74": 11, "75-79": 12, "80+": 13
        },
        "Education": {
            "Never attended school": 1,
            "Grade 1-8": 2,
            "Some high school": 3,
            "High school graduate": 4,
            "Some college": 5,
            "College graduate": 6
        },
        "Income": {
            "<10k": 1, "10-15k": 2, "15-20k": 3, "20-25k": 4,
            "25-35k": 5, "35-50k": 6, "50-75k": 7, ">75k": 8
        }
    }

    ordered_fields = [
        "HighBP", "HighChol", "CholCheck", "BMI", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "GenHlth",
        "MentHlth", "PhysHlth", "DiffWalk", "Sex", "Age", "Education",
        "Income"
    ]

    numeric_list = []
    for field in ordered_fields:
        val = form.get(field)
        if val is None:
            raise ValueError(f"Missing input for field: {field}")

        if field in mapping:
            if val not in mapping[field]:
                raise ValueError(f"Invalid value '{val}' for field '{field}'")
            val = mapping[field][val]

        try:
            numeric_list.append(float(val))
        except ValueError:
            raise ValueError(f"Cannot convert value '{val}' of field '{field}' to float")

    return numeric_list

