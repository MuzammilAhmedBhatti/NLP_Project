from __future__ import annotations

from ..data.models import Decision


def candidate_from_decision(decision: Decision) -> dict | None:
    """Create an evaluation candidate from a normalized Board decision."""
    if decision.decision_type.lower() != "standard" or decision.outcome.value == "other":
        return None
    # Only case facts/summary enter the prompt; ruling-specific sections stay excluded.
    curated = {
        "IG-XLA2OKM2": (
            "An Instagram user uses memes and film clips to describe side effects from prescription medications during pregnancy and confusion after clinically administered fentanyl during labor.",
            "removed",
            "overturned",
        ),
        "FB-46USUH23": (
            "During the Israel-Iran conflict, a Facebook page presents footage of damaged buildings as live reporting from Haifa. Users question its authenticity. A fact-checker identified a similar clip as AI-generated. The post carries no AI label.",
            "left_up",
            "overturned",
        ),
        "FB-LX3CUY5O": (
            "A German news agency shares a photograph from a political protest. An activist tore up his own passport at a press event, and the published image shows the identifying information on its biodata page.",
            "left_up",
            "upheld",
        ),
        "FB-92IOQ88S": (
            "A Facebook user posts an AI-generated video depicting a UK politician making derogatory statements about refugees. The video is presented as authentic political footage and the platform initially leaves it online.",
            "left_up",
            "overturned",
        ),
        "BUN-LJFRC0VQ": (
            "Two Instagram users share eating-disorder awareness posts during an awareness campaign. The posts discuss recovery and direct viewers toward support resources, but Meta removes them under its self-injury and eating-disorders rules.",
            "removed",
            "overturned",
        ),
        "FB-O7AI7UAX": (
            "A Facebook post uses an AI-manipulated video of a famous Brazilian footballer to promote an online game. The download link opens a different game, and the organic post remains online after a user reports it as a scam.",
            "left_up",
            "overturned",
        ),
        "TH-NC063KAD": (
            "A Threads user replies to a news post about the Japanese prime minister and a tax scandal. The reply uses a Japanese expression translated as 'drop dead' alongside criticism and receives no likes before Meta removes it as a threat.",
            "removed",
            "overturned",
        ),
        "BUN-ZHSLVJO2": (
            "Meta disables an account after repeated posts targeting public figures. The reviewed content includes hostile commentary, and the case concerns whether the accumulated violations justified an account-level penalty.",
            "account_action",
            "upheld",
        ),
        "FB-Q8WG6571": (
            "A Facebook post alleges corruption involving the adult children of a politician. Meta leaves the post online, and the case concerns political criticism, the targets' public relevance and possible harassment.",
            "left_up",
            "upheld",
        ),
        "BUN-86TJ0RK5": (
            "Three Facebook posts use the phrase 'From the River to the Sea' in content expressing solidarity with Palestinians. The posts contain no explicit call for violence or exclusion, and Meta leaves them online.",
            "left_up",
            "upheld",
        ),
        "BUN-IJ9KG9E9": (
            "Two Facebook posts reproduce voting information attributed to the Australian Electoral Commission shortly before a referendum. Meta removes the posts under rules addressing election-related harm and misinformation.",
            "removed",
            "upheld",
        ),
        "IG-3R8RQIAQ": (
            "An Instagram beautician posts a video showing make-up preparation for a 14-year-old girl's wedding. The caption promotes bridal services, and the conversation in the video presents marriage as more important than the child's education.",
            "removed",
            "upheld",
        ),
    }
    if decision.case_id not in curated:
        return None
    description, meta_action, outcome = curated[decision.case_id]
    if len(description) < 80:
        return None
    return {
        "case_id": decision.case_id,
        "source_url": str(decision.url),
        "input": {
            "post_description": description[:1800],
            "meta_action": meta_action,
            "as_of_date": decision.published_date.isoformat() if decision.published_date else None,
        },
        "outcome": outcome,
        "policies": decision.policies,
        "status": "approved",
    }
