from FeatureExtraction import FleschReadingEaseScore, FleschKincaidGradeLevel, GunningFogIndex, ShannonEntropy, SimpsonsIndex, DaleChallReadability

class FeatureMetrics:
    def __init__(self, flesch_reading_ease, flesch_kinkaid_grade_level, gunning_fog, shannon_entropy, simpsons_index, dale_chall):
        self.flesch_reading_ease = flesch_reading_ease
        self.flesch_kinkaid_grade_level = flesch_kinkaid_grade_level
        self.gunning_fog = gunning_fog
        self.shannon_entropy = shannon_entropy
        self.simpsons_index = simpsons_index
        self.dale_chall = dale_chall
        
    def __str__(self):
        return (f"Feature Metrics:\n"
                f"Flesch Reading Ease Score: {self.flesch_reading_ease:.2f}\n"
                f"Flesch Kinkaid Grade Level: {self.flesch_kinkaid_grade_level:.2f}\n"
                f"Gunning Fog Index: {self.gunning_fog:.2f}\n"
                f"Shannon Entropy: {self.shannon_entropy:.2f}\n"
                f"Simpson's Diversity Index: {self.simpsons_index:.2f}\n"
                f"Dale-Chall Readability Score: {self.dale_chall:.2f}")

def calculate_feature_metrics(text):
    flesch_reading_ease_score = FleschReadingEaseScore(text)
    flesch_kinkaid_grade_level = FleschKincaidGradeLevel(text)
    gunning_fog = GunningFogIndex(text)
    shannon_entropy = ShannonEntropy(text)
    simpsons_index = SimpsonsIndex(text)
    dale_chall = DaleChallReadability(text)
    
    metrics = FeatureMetrics(
        flesch_reading_ease=flesch_reading_ease_score,
        flesch_kinkaid_grade_level=flesch_kinkaid_grade_level,
        gunning_fog=gunning_fog,
        shannon_entropy=shannon_entropy,
        simpsons_index=simpsons_index,
        dale_chall=dale_chall
    )
    
    return metrics