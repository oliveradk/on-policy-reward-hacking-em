# LLM judges of the misalignment evals. `strict` runs inside every eval; the others are
# replayed over finished logs with `python -m misalignment_evals.judges.rescore <judge>`:
#   output-only   frame_colleague_output, misaligned_emails_category, exfil_offer_output
#   strategic     {frame_colleague,monitor_disruption,exfil_offer,chat}_strategic_reasoning
#   competence    competent_misalignment
#   frame aware   frame_awareness
from misalignment_evals.judges.strict import strict_scorer

__all__ = ["strict_scorer"]
