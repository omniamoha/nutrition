# ===============================
# Full Hormone & Endocrine Config
# Smart Nutrition AI System
# ===============================

HORMONE_CONFIG = {

    # ===========================
    # 1. Insulin
    # ===========================
    "Insulin": {

        "score_name": "Insulin_Score",

        "ranges": {
            "Infant (0-2)": (2, 15),
            "Child (3-12)": (3, 20),
            "Teen (13-18)": (3, 25),
            "Adult (19-50)": {
                "Male": (2, 25),
                "Female": (3, 27)
            },
            "Senior (50+)": (2, 30)
        },

        "default_range": (2, 25),

        "scoring": {

            "High": [
                {"column": "fiber", "weight": 0.6},
                {"column": "sugars", "weight": -0.9},
                {"column": "magnesium", "weight": 0.3}
            ],

            "Low": [
                {"column": "protein", "weight": 0.5},
                {"column": "carbohydrate", "weight": 0.4}
            ],

            "Normal": [
                {"column": "fiber", "weight": 0.3},
                {"column": "protein", "weight": 0.3}
            ]
        },

        "advice": {
            "High": "Reduce sugar intake and increase fiber.",
            "Low": "Increase balanced carbohydrates and protein.",
            "Normal": "Maintain balanced diet."
        }
    },


    # ===========================
    # 2. Thyroid (TSH)
    # ===========================
    "TSH": {

        "score_name": "Thyroid_Score",
         

        "ranges":{
            "Infant (0-2)": (0.7, 6),
            "Child (3-12)": (0.7, 5.5),
            "Teen (13-18)": (0.5, 4.5),
            "Adult (19-50)": {
                "Male": (0.4, 4.5),
                "Female": (0.4, 4.5)
            },
            "Senior (50+)": (0.5, 5)
       
        },

        "default_range": (0.4, 4.0),

        "scoring": {

            "High": [
                {"column": "calcium", "weight": 0.3}
            ],

            "Low": [
                {"column": "selenium", "weight": 0.6},
                {"column": "zinc", "weight": 0.4},
                {"column": "iodine", "weight": 0.5}
            ],

            "Normal": [
                {"column": "vitamin_d", "weight": 0.4}
            ]
        },

        "advice": {
            "High": "Possible hypothyroidism risk.",
            "Low": "Possible hyperthyroidism risk.",
            "Normal": "Maintain thyroid-supporting diet."
        }
    },

    
"T3": {

    "score_name": "T3_Score",

    "ranges": {

        "Infant (0-2)": {
            "Male": (100, 300),
            "Female": (100, 300)
        },

        "Child (3-12)": {
            "Male": (100, 250),
            "Female": (100, 250)
        },

        "Teen (13-18)": {
            "Male": (90, 200),
            "Female": (90, 200)
        },

        "Adult (19-50)": {
            "Male": (80, 200),
            "Female": (80, 200)
        
        },

        "Senior (50+)": {
            "Male": (70, 180),
            "Female": (70, 180)
        }
    },

    "default_range": (80, 200),

    "scoring": {

        "Low": [
            {"column": "selenium", "weight": 0.6},
            {"column": "iodine", "weight": 0.5},
            {"column": "zinc", "weight": 0.4}
        ],

        "High": [
            {"column": "fiber", "weight": 0.3}
        ],

        "Normal": [
            {"column": "protein", "weight": 0.2}
        ]
    },

    "advice": {

        "Low": "Support thyroid function with selenium, iodine and zinc.",

        "High": "Avoid excessive iodine intake and monitor thyroid function.",

        "Normal": "Maintain a balanced thyroid-supportive diet."
    }
},


"Free_T4": {

    "score_name": "Free_T4_Score",

    "ranges": {

        "Infant (0-2)": {
            "Male": (0.8, 2.2),
            "Female": (0.8, 2.2)
        },

        "Child (3-12)": {
            "Male": (0.8, 2.0),
            "Female": (0.8, 2.0)
        },

        "Teen (13-18)": {
            "Male": (0.8, 1.8),
            "Female": (0.8, 1.8)
        },

        "Adult (19-50)": {
            "Male": (0.8, 1.8),
            "Female": (0.8, 1.8)
        
        },

        "Senior (50+)": {
            "Male": (0.7, 1.7),
            "Female": (0.7, 1.7)
        }
    },

    "default_range": (0.8, 1.8),

    "scoring": {

        "Low": [
            {"column": "selenium", "weight": 0.6},
            {"column": "iodine", "weight": 0.5},
            {"column": "zinc", "weight": 0.4}
        ],

        "High": [
            {"column": "fiber", "weight": 0.3}
        ],

        "Normal": [
            {"column": "protein", "weight": 0.2}
        ]
    },

    "advice": {

        "Low": "Support thyroid hormone production with adequate iodine and selenium intake.",

        "High": "Monitor thyroid function and avoid excessive iodine supplementation.",

        "Normal": "Maintain a balanced thyroid-friendly diet."
    }
},


    # ===========================
    # 2. LIVER
    # ===========================
"ALT": {

    "score_name": "ALT_Score",

    "ranges": {

        "Infant (0-2)": {
            "Male": (10, 40),
            "Female": (10, 40)
        },

        "Child (3-12)": {
            "Male": (10, 40),
            "Female": (10, 40)
        },

        "Teen (13-18)": {
            "Male": (10, 45),
            "Female": (10, 40)
        },

        "Adult (19-50)": {
            "Male": (7, 56),
            "Female": (7, 35)
        },

        "Senior (50+)": {
            "Male": (7, 50),
            "Female": (7, 32)
        }
    },

    "default_range": (7, 56),

    "scoring": {

        "Low": [
            {"column": "protein", "weight": 0.2}
        ],

        "High": [
            {"column": "fiber", "weight": 0.4},
            {"column": "vitamin_c", "weight": 0.4},
            {"column": "vitamin_e", "weight": 0.5}
        ],

        "Normal": [
            {"column": "fiber", "weight": 0.2}
        ]
    },

    "advice": {

        "Low": "Usually not clinically significant.",

        "High": "Support liver health with antioxidants, fruits and vegetables.",

        "Normal": "Maintain a liver-friendly balanced diet."
    }
},


"AST": {

    "score_name": "AST_Score",

    "ranges": {

        "Infant (0-2)": {
            "Male": (15, 50),
            "Female": (15, 50)
        },

        "Child (3-12)": {
            "Male": (15, 45),
            "Female": (15, 45)
        },

        "Teen (13-18)": {
            "Male": (10, 40),
            "Female": (10, 40)
        },

        "Adult (19-50)": {
            "Male": (10, 40),
            "Female": (9, 32)
        },

        "Senior (50+)": {
            "Male": (10, 38),
            "Female": (9, 30)
        }
    },

    "default_range": (10, 40),

    "scoring": {

        "Low": [
            {"column": "protein", "weight": 0.2}
        ],

        "High": [
            {"column": "fiber", "weight": 0.4},
            {"column": "vitamin_c", "weight": 0.4},
            {"column": "vitamin_e", "weight": 0.5}
        ],

        "Normal": [
            {"column": "fiber", "weight": 0.2}
        ]
    },

    "advice": {

        "Low": "Usually not clinically significant.",

        "High": "Elevated AST may indicate liver or muscle stress. Support recovery with antioxidant-rich foods.",

        "Normal": "Maintain a healthy balanced diet."
    }
},



    # ===========================
    # 4. KIDNEY
    # ===========================
"Urea": {

    "score_name": "Urea_Score",

    "ranges": {

        "Infant (0-2)": {
            "Male": (5, 18),
            "Female": (5, 18)
        },

        "Child (3-12)": {
            "Male": (7, 20),
            "Female": (7, 20)
        },

        "Teen (13-18)": {
            "Male": (10, 25),
            "Female": (10, 25)
        },

        "Adult (19-50)": {
            "Male": (15, 40),
            "Female": (15, 40)
        },

        "Senior (50+)": {
            "Male": (15, 45),
            "Female": (15, 45)
        }
    },

    "default_range": (15, 40),

    "scoring": {

        "Low": [
            {"column": "protein", "weight": 0.4}
        ],

        "High": [
            {"column": "water", "weight": 0.6},
            {"column": "fiber", "weight": 0.3}
        ],

        "Normal": [
            {"column": "protein", "weight": 0.2}
        ]
    },

    "advice": {

        "Low": "Ensure adequate protein intake.",

        "High": "Increase hydration and moderate excessive protein intake.",

        "Normal": "Maintain balanced nutrition and hydration."
    }
},


"Creatinine": {

    "score_name": "Creatinine_Score",

    "ranges": {

        "Infant (0-2)": {
            "Male": (0.2, 0.5),
            "Female": (0.2, 0.5)
        },

        "Child (3-12)": {
            "Male": (0.3, 0.7),
            "Female": (0.3, 0.7)
        },

        "Teen (13-18)": {
            "Male": (0.5, 1.0),
            "Female": (0.5, 0.9)
        },

        "Adult (19-50)": {
            "Male": (0.7, 1.3),
            "Female": (0.5, 1.1)
        },

        "Senior (50+)": {
            "Male": (0.7, 1.4),
            "Female": (0.5, 1.2)
        }
    },

    "default_range": (0.7, 1.3),

    "scoring": {

        "Low": [
            {"column": "protein", "weight": 0.3}
        ],

        "High": [
            {"column": "water", "weight": 0.7},
            {"column": "fiber", "weight": 0.3}
        ],

        "Normal": [
            {"column": "protein", "weight": 0.2}
        ]
    },

    "advice": {

        "Low": "Maintain adequate protein intake and muscle health.",

        "High": "Support kidney health through hydration and balanced nutrition.",

        "Normal": "Maintain healthy kidney-supportive habits."
    }
},
}