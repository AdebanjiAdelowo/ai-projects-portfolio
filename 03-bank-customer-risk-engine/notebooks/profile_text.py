"""Canonical text representations of customer and transaction records.

The sentence-transformer embeddings the risk models are trained on are computed from these strings
(notebooks 01 and 02), and the inference demo in notebook 03 must embed exactly the same strings, so
all three notebooks import these functions instead of formatting text themselves.

Every field listed in CUSTOMER_FIELDS / TRANSACTION_FIELDS is required. A missing or NaN field raises
ValueError rather than being filled with a default, so training and inference cannot silently disagree.
"""

import math

CUSTOMER_FIELDS = (
    "age", "employment_type", "annual_income", "credit_score", "account_balance",
    "num_credit_cards", "employment_years", "num_prev_loans", "previous_default",
    "monthly_expenses", "num_dependents", "debt_to_income",
)
TRANSACTION_FIELDS = (
    "product_type", "amount", "term_months", "interest_rate", "collateral", "risk_level",
)


def _require(record, fields):
    missing = []
    for name in fields:
        value = record[name] if name in record else None
        if value is None or (isinstance(value, float) and math.isnan(value)):
            missing.append(name)
    if missing:
        raise ValueError("missing required field(s): " + ", ".join(missing))


def credit_label(score):
    if score < 580:  return 'Very Poor'
    if score < 670:  return 'Fair'
    if score < 740:  return 'Good'
    if score < 800:  return 'Very Good'
    return 'Exceptional'


def customer_to_text(row):
    """Customer record (dict or pandas row) -> the text the client embedding is computed from."""
    _require(row, CUSTOMER_FIELDS)
    default_note = (
        'Has a previous default on record.'
        if row['previous_default'] == 1
        else 'No previous defaults.'
    )
    parts = [
        'Customer profile: {}-year-old {}.'.format(int(row['age']), row['employment_type'].lower()),
        'Annual income: ${:,}.'.format(int(row['annual_income'])),
        'Credit score: {} ({}).'.format(int(row['credit_score']), credit_label(int(row['credit_score']))),
        'Account balance: ${:,}.'.format(int(row['account_balance'])),
        'Credit cards held: {}.'.format(int(row['num_credit_cards'])),
        'Employment duration: {} years.'.format(int(row['employment_years'])),
        'Previous loans taken: {}.'.format(int(row['num_prev_loans'])),
        default_note,
        'Monthly expenses: ${:,}.'.format(int(row['monthly_expenses'])),
        'Dependents: {}.'.format(int(row['num_dependents'])),
        'Debt-to-income ratio: {:.2f}.'.format(float(row['debt_to_income'])),
    ]
    return ' '.join(parts)


def transaction_to_text(row):
    """Transaction record (dict or pandas row) -> the text the transaction embedding is computed from."""
    _require(row, TRANSACTION_FIELDS)
    collateral_note = 'Collateral required.' if row['collateral'] else 'No collateral required.'
    parts = [
        'Product type: {}.'.format(row['product_type']),
        'Loan amount requested: ${:,}.'.format(int(row['amount'])),
        'Repayment term: {} months.'.format(int(row['term_months'])),
        'Interest rate: {:.2f}%.'.format(float(row['interest_rate'])),
        collateral_note,
        'Inherent risk level of product: {}.'.format(row['risk_level']),
        'Monthly instalment estimate: ${:,.0f}.'.format(
            float(row['amount']) * float(row['interest_rate']) / 100 / 12
            + float(row['amount']) / max(int(row['term_months']), 1)
        ),
    ]
    return ' '.join(parts)
