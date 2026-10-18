
def analyze_threat(row, ai_prediction):
    """
    Interpret a traffic record using example expert rules.

    ai_prediction:
        0 = benign prediction
        1 = attack prediction

    These rules are illustrative and must be validated
    before use in a real security environment.
    """

    packets_per_second = float(row.get("Flow Packets/s", 0) or 0)
    syn_flags = float(row.get("SYN Flag Count", 0) or 0)

    # Rule 1: AI has flagged the record as an attack.
    if ai_prediction == 1:

        # Rule 2: High packet rate may indicate a traffic flood.
        if packets_per_second > 1000:
            return {
                "threat": "Possible Traffic Flood",
                "risk": "High",
                "reason": (
                    "The AI flagged the traffic and its packet rate "
                    "exceeds the example rule threshold."
                ),
                "action": (
                    "Review traffic sources, flow logs, and "
                    "network protection settings."
                ),
            }

        # Rule 3: Many SYN flags may indicate scan-like activity.
        if syn_flags >= 5:
            return {
                "threat": "Suspicious SYN Activity",
                "risk": "High",
                "reason": (
                    "The AI flagged the traffic and the SYN flag "
                    "count meets the example rule threshold."
                ),
                "action": (
                    "Review connection patterns and restrict "
                    "unnecessary exposed services if appropriate."
                ),
            }

        # Rule 4: Attack prediction without a specific rule match.
        return {
            "threat": "Suspicious Network Traffic",
            "risk": "Medium",
            "reason": (
                "The AI classified the record as an attack, "
                "but no more specific expert rule matched."
            ),
            "action": (
                "Inspect the record and related logs before "
                "taking defensive action."
            ),
        }

    # Rule 5: No attack predicted by the model.
    return {
        "threat": "No Attack Predicted",
        "risk": "Low",
        "reason": (
            "The AI classified the record as benign and no "
            "additional expert rule was triggered."
        ),
        "action": (
            "Continue routine monitoring. This result does "
            "not guarantee that the traffic is safe."
        ),
    }
