"""Tests for DealDesk MCP server."""
from __future__ import annotations
import json
import asyncio
import pytest
from dealdesk_mcp.server import (
    mcp,
    dealdesk_run_waterfall, WaterfallInput,
    dealdesk_fund_metrics, MetricsInput,
    dealdesk_price_secondary, SecondaryInput,
    dealdesk_continuation_vehicle, CVInput,
    dealdesk_simulate_exit, ExitInput, FounderInput, RoundInput,
    dealdesk_generate_teaser, TeaserInput, TeaserFinancialInput,
    ResponseFormat,
)


def run(coro):
    return asyncio.run(coro)


# ── Server registration ───────────────────────────────────────────────────────

def test_server_name():
    assert mcp.name == "dealdesk_mcp"

def test_all_tools_registered():
    tools = run(mcp.list_tools())
    names = {t.name for t in tools}
    expected = {
        "dealdesk_run_waterfall", "dealdesk_fund_metrics",
        "dealdesk_price_secondary", "dealdesk_continuation_vehicle",
        "dealdesk_simulate_exit", "dealdesk_generate_teaser",
    }
    assert expected.issubset(names)

def test_tools_have_descriptions():
    tools = run(mcp.list_tools())
    for t in tools:
        assert t.description and len(t.description) > 20

def test_tools_have_input_schema():
    tools = run(mcp.list_tools())
    for t in tools:
        assert t.inputSchema is not None


# ── Waterfall tool ────────────────────────────────────────────────────────────

def test_waterfall_json():
    out = run(dealdesk_run_waterfall(WaterfallInput(
        total_proceeds=250, committed_capital=100,
        response_format=ResponseFormat.JSON)))
    data = json.loads(out)
    assert data["structure"] == "European"
    assert abs(data["total_proceeds"] - 250) < 1
    assert data["gp_carry"] > 0

def test_waterfall_markdown():
    out = run(dealdesk_run_waterfall(WaterfallInput(
        total_proceeds=250, committed_capital=100)))
    assert "Waterfall" in out
    assert "LP MOIC" in out

def test_waterfall_american():
    out = run(dealdesk_run_waterfall(WaterfallInput(
        total_proceeds=200, committed_capital=100, structure="american",
        response_format=ResponseFormat.JSON)))
    data = json.loads(out)
    assert data["structure"] == "American"

def test_waterfall_reconciles():
    out = run(dealdesk_run_waterfall(WaterfallInput(
        total_proceeds=300, committed_capital=100, gp_commitment_pct=0.02,
        response_format=ResponseFormat.JSON)))
    data = json.loads(out)
    assert abs(data["lp_distribution"] + data["gp_distribution"] - 300) < 1


# ── Metrics tool ──────────────────────────────────────────────────────────────

def test_metrics_irr_moic():
    out = run(dealdesk_fund_metrics(MetricsInput(cashflows=[-100, 0, 0, 250])))
    data = json.loads(out)
    assert data["moic"] == pytest.approx(2.5)
    assert data["irr"] is not None and data["irr"] > 0

def test_metrics_dpi():
    out = run(dealdesk_fund_metrics(MetricsInput(cashflows=[-100, 80])))
    data = json.loads(out)
    assert data["dpi"] == pytest.approx(0.8)

def test_metrics_with_residual():
    out = run(dealdesk_fund_metrics(MetricsInput(cashflows=[-100, 50], residual_nav=100)))
    data = json.loads(out)
    assert data["tvpi"] == pytest.approx(1.5)


# ── Secondary tool ────────────────────────────────────────────────────────────

def test_secondary_discount():
    out = run(dealdesk_price_secondary(SecondaryInput(nav=100, bid_pct_nav=0.85)))
    data = json.loads(out)
    assert data["price"] == 85.0
    assert data["is_discount"] is True

def test_secondary_premium():
    out = run(dealdesk_price_secondary(SecondaryInput(nav=100, bid_pct_nav=1.10)))
    data = json.loads(out)
    assert data["is_discount"] is False


# ── Continuation vehicle tool ─────────────────────────────────────────────────

def test_cv_crystallized_carry():
    out = run(dealdesk_continuation_vehicle(CVInput(
        asset_nav=100, purchase_price=110, asset_cost_basis=40, rollover_pct=0.6)))
    data = json.loads(out)
    assert data["crystallized_carry"] > 0
    assert data["premium_to_nav"] == pytest.approx(0.10)

def test_cv_no_gain():
    out = run(dealdesk_continuation_vehicle(CVInput(
        asset_nav=100, purchase_price=90, asset_cost_basis=100, rollover_pct=0.5)))
    data = json.loads(out)
    assert data["crystallized_carry"] == 0.0


# ── Exit simulation tool ──────────────────────────────────────────────────────

def test_exit_common_only():
    out = run(dealdesk_simulate_exit(ExitInput(
        founders=[FounderInput(name="Alice", shares=6_000_000),
                  FounderInput(name="Bob", shares=4_000_000)],
        exit_value=100_000_000)))
    data = json.loads(out)
    alice = next(p for p in data["payouts"] if p["name"] == "Alice")
    assert alice["proceeds"] == pytest.approx(60_000_000)

def test_exit_with_liq_pref():
    out = run(dealdesk_simulate_exit(ExitInput(
        founders=[FounderInput(name="Alice", shares=8_000_000)],
        rounds=[RoundInput(name="Seed", pre_money=8e6, investment=2e6,
                           investor="Seed Fund", liq_pref=1.0)],
        exit_value=4_000_000)))
    data = json.loads(out)
    seed = next(p for p in data["payouts"] if p["name"] == "Seed Fund")
    assert seed["proceeds"] == pytest.approx(2_000_000)  # took 1x pref
    assert seed["took_preference"]

def test_exit_reconciles():
    out = run(dealdesk_simulate_exit(ExitInput(
        founders=[FounderInput(name="A", shares=8_000_000)],
        esop=1_000_000,
        rounds=[RoundInput(name="Seed", pre_money=90e6, investment=10e6,
                           investor="Fund")],
        exit_value=1_000_000_000)))
    data = json.loads(out)
    total = sum(p["proceeds"] for p in data["payouts"])
    assert abs(total - 1_000_000_000) < 100


# ── Teaser tool ───────────────────────────────────────────────────────────────

def test_teaser_basic():
    out = run(dealdesk_generate_teaser(TeaserInput(
        name="Acme Robotics", sector="Robotics", ask_amount_cr=150,
        highlights=["50+ clients"],
        financials=[TeaserFinancialInput(year="FY2023", revenue=70, ebitda=10),
                    TeaserFinancialInput(year="FY2024", revenue=120, ebitda=24)])))
    assert "Acme Robotics" in out
    assert "CAGR" in out

def test_teaser_anonymized():
    out = run(dealdesk_generate_teaser(TeaserInput(
        name="Acme Robotics", sector="Robotics", code_name="Project Atlas",
        anonymize=True)))
    assert "Acme Robotics" not in out
    assert "Project Atlas" in out


# ── Error handling ────────────────────────────────────────────────────────────

def test_waterfall_handles_bad_input_gracefully():
    # carry >= 1 should be caught by FundTerms → returns Error string, not crash
    # (Pydantic blocks carry>=1 at input; test a runtime edge instead)
    out = run(dealdesk_run_waterfall(WaterfallInput(
        total_proceeds=100, committed_capital=100,
        response_format=ResponseFormat.JSON)))
    data = json.loads(out)
    assert data["gp_carry"] == 0.0  # proceeds == capital, no profit
