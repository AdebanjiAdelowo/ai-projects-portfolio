"""Train/serve consistency of the text that customer and transaction embeddings are computed from.

Run from the project directory:  pytest tests/
"""
import json
import math
import re
import sys
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from profile_text import (  # noqa: E402
    CUSTOMER_FIELDS,
    TRANSACTION_FIELDS,
    credit_label,
    customer_to_text,
    transaction_to_text,
)

CUSTOMER = {  # Demo 1 in notebook 03
    "age": 40, "annual_income": 120_000, "credit_score": 780, "account_balance": 25_000,
    "num_credit_cards": 2, "employment_years": 12, "employment_type": "Employed",
    "num_prev_loans": 3, "previous_default": 0, "monthly_expenses": 3_500,
    "num_dependents": 2, "debt_to_income": 0.35,
}
TRANSACTION = {
    "product_type": "Personal Loan", "amount": 15_000, "term_months": 36,
    "interest_rate": 8.5, "collateral": False, "risk_level": "Medium",
}

# The exact strings notebooks 01 and 02 embedded for training before the refactor; the stored
# training embeddings were computed from this format, so it must not drift.
CUSTOMER_TEXT = (
    "Customer profile: 40-year-old employed. Annual income: $120,000. Credit score: 780 (Very Good). "
    "Account balance: $25,000. Credit cards held: 2. Employment duration: 12 years. "
    "Previous loans taken: 3. No previous defaults. Monthly expenses: $3,500. Dependents: 2. "
    "Debt-to-income ratio: 0.35."
)
TRANSACTION_TEXT = (
    "Product type: Personal Loan. Loan amount requested: $15,000. Repayment term: 36 months. "
    "Interest rate: 8.50%. No collateral required. Inherent risk level of product: Medium. "
    "Monthly instalment estimate: $523."
)


def test_training_format_is_unchanged():
    assert customer_to_text(CUSTOMER) == CUSTOMER_TEXT
    assert transaction_to_text(TRANSACTION) == TRANSACTION_TEXT


def _code(nb_name):
    cells = json.loads((NOTEBOOKS / nb_name).read_text())["cells"]
    return "\n".join("".join(c["source"]) for c in cells if c["cell_type"] == "code")


def test_training_and_inference_notebooks_use_the_canonical_functions():
    train_c, train_t, infer = (_code("01_Client_Embeddings.ipynb"), _code("02_Product_Embeddings.ipynb"),
                               _code("03_Risk_Decision_Model.ipynb"))
    assert "from profile_text import customer_to_text" in train_c
    assert "customers_df.apply(customer_to_text" in train_c
    assert "from profile_text import transaction_to_text" in train_t
    assert "txn_df.apply(transaction_to_text" in train_t
    assert "from profile_text import customer_to_text, transaction_to_text" in infer
    assert "encode([customer_to_text(profile)])" in infer
    assert "encode([transaction_to_text(txn)])" in infer
    # no notebook defines its own customer/transaction text format any more
    for code in (train_c, train_t, infer):
        assert not re.search(r"def (customer_to_text|product_to_text|credit_label)\b", code)
        assert "Credit cards:" not in code and "Loan amount:" not in code


def test_categorical_descriptions_are_preserved():
    assert "(Very Good)" in customer_to_text(CUSTOMER)
    assert "Has a previous default on record." in customer_to_text({**CUSTOMER, "previous_default": 1})
    assert "40-year-old self-employed." in customer_to_text({**CUSTOMER, "employment_type": "Self-employed"})
    assert "Collateral required." in transaction_to_text({**TRANSACTION, "collateral": True})
    assert "Inherent risk level of product: High." in transaction_to_text({**TRANSACTION, "risk_level": "High"})
    assert [credit_label(s) for s in (579, 580, 669, 670, 739, 740, 799, 800)] == [
        "Very Poor", "Fair", "Fair", "Good", "Good", "Very Good", "Very Good", "Exceptional"]


def test_field_order_is_deterministic():
    shuffled = dict(reversed(list(CUSTOMER.items())))
    assert customer_to_text(shuffled) == CUSTOMER_TEXT
    assert transaction_to_text(dict(reversed(list(TRANSACTION.items())))) == TRANSACTION_TEXT
    order = ["Customer profile", "Annual income", "Credit score", "Account balance", "Credit cards held",
             "Employment duration", "Previous loans taken", "previous default", "Monthly expenses",
             "Dependents", "Debt-to-income ratio"]
    positions = [CUSTOMER_TEXT.index(label) for label in order]
    assert positions == sorted(positions)


@pytest.mark.parametrize("field", CUSTOMER_FIELDS)
@pytest.mark.parametrize("bad", ["missing", None, math.nan])
def test_missing_customer_fields_raise(field, bad):
    record = dict(CUSTOMER)
    if bad == "missing":
        del record[field]
    else:
        record[field] = bad
    with pytest.raises(ValueError, match=field):
        customer_to_text(record)


@pytest.mark.parametrize("field", TRANSACTION_FIELDS)
def test_missing_transaction_fields_raise(field):
    record = {k: v for k, v in TRANSACTION.items() if k != field}
    with pytest.raises(ValueError, match=field):
        transaction_to_text(record)


def test_false_and_zero_are_values_not_missing():
    # collateral=False and previous_default=0 are legitimate values and must not be treated as missing
    assert "No collateral required." in transaction_to_text(TRANSACTION)
    assert "No previous defaults." in customer_to_text(CUSTOMER)
