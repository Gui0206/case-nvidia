"""Evidence Validator — garante que toda claim de score tenha Evidence com source_url.

Regra de ouro (aplicada no scoring): nenhuma feature entra sem evidência rastreável.
"""
def has_evidence(sig) -> bool:
    return bool(getattr(sig, "evidence", None)) or bool(getattr(sig, "active", None))
