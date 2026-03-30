"""
DailyDialog act and emotion code → label mapping (single source of truth).

Dataset: https://huggingface.co/datasets/roskoN/dailydialog
Paper: Li et al. 2017, DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset.
"""

# Dialog act: communicative intention of the utterance (index = code from dataset).
# -1 or missing = unlabeled.
ACT_LABELS = [
    "inform",      # 0: stating information
    "question",    # 1: asking a question
    "directive",   # 2: request, command, suggestion
    "commissive",  # 3: promise, offer, commitment
]

# Emotion: emotion label for the utterance (index = code from dataset).
# -1 or missing = unlabeled.
EMOTION_LABELS = [
    "no emotion",  # 0
    "anger",       # 1
    "disgust",     # 2
    "fear",        # 3
    "happiness",   # 4
    "sadness",     # 5
    "surprise",    # 6
]
