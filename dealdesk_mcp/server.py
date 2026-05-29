#!/usr/bin/env python3
"""
DealDesk MCP — expose CarryFlow / ExitSim / TeaserGen deal tools inside Claude,
ChatGPT, or any MCP client.

Lets an LLM run a PE/VC distribution waterfall, simulate a cap-table exit, price
a secondary, model a continuation vehicle, compute fund metrics, or generate an
investor teaser — all by chatting.

Wraps these libraries (https://github.com/bhupendra05):
  carryflow · exitsim · teasergen
"""
from __future__ import annotations
import json
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# Deal-tool libraries
from carryflow import FundTerms, european_waterfall, american_waterfall
from carryflow.metrics import xirr, irr, moic, dpi, tvpi
from carryflow.secondary import price_secondary, model_continuation_vehicle
from exitsim import CapTable, FundingRound, simulate_exit
from teasergen import CompanyInfo, FinancialYear, generate_teaser, render_markdown

# India market tools (optional — degrade gracefully if not installed)
try:
    from india_comps.fetch import fetch_company as _fetch_comps_company
    from india_comps import build_comps as _build_comps
    from india_comps.comps import implied_value as _implied_value, DEFAULT_METRICS as _COMPS_METRICS
    _HAS_COMPS = True
except Exception:
    _HAS_COMPS = False

try:
    from india_dcf.fetch import fetch_india_financials as _fetch_dcf
    from india_dcf import (run_india_dcf as _run_dcf, calculate_wacc as _calc_wacc,
                           IndiaWACCParams as _WACCParams, IndiaDCFAssumptions as _DCFAssump,
                           SECTOR_BETA as _SECTOR_BETA)
    _HAS_DCF = True
except Exception:
    _HAS_DCF = False

try:
    from drhp_intel import analyze as _drhp_analyze
    from drhp_intel.parser import read_pdf as _read_pdf
    _HAS_DRHP = True
except Exception:
    _HAS_DRHP = False

mcp = FastMCP("dealdesk_mcp")


# ── Shared helpers ────────────────────────────────────────────────────────────

class ResponseFormat(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"


def _err(e: Exception) -> str:
    return f"Error: {type(e).__name__}: {e}"


# ════════════════════════════════════════════════════════════════════════════
# Tool 1 — Distribution Waterfall
# ════════════════════════════════════════════════════════════════════════════

class WaterfallInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    total_proceeds: float = Field(..., description="Total proceeds to distribute (same unit as committed, e.g. ₹ Crore)", gt=0)
    committed_capital: float = Field(..., description="Total committed capital (LP + GP)", gt=0)
    gp_commitment_pct: float = Field(0.02, description="GP co-investment as fraction (0.02 = 2%)", ge=0, lt=1)
    preferred_return: float = Field(0.08, description="Preferred return / hurdle rate (0.08 = 8%)", ge=0, lt=1)
    carry: float = Field(0.20, description="Carried interest fraction (0.20 = 20%)", ge=0, lt=1)
    years: float = Field(5.0, description="Holding period in years (drives compounded preferred return)", gt=0)
    structure: str = Field("european", description="'european' (whole-fund) or 'american' (deal-by-deal)")
    response_format: ResponseFormat = Field(ResponseFormat.MARKDOWN, description="'markdown' or 'json'")


@mcp.tool(
    name="dealdesk_run_waterfall",
    annotations={"title": "Run PE/VC Distribution Waterfall", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def dealdesk_run_waterfall(params: WaterfallInput) -> str:
    """Run a PE/VC distribution waterfall and compute LP/GP splits and carry.

    Distributes total_proceeds through the standard 4-tier waterfall:
    (1) return of capital, (2) preferred return, (3) GP catch-up, (4) carry split.

    Args:
        params (WaterfallInput): committed capital, proceeds, pref, carry, years,
            and structure ('european' whole-fund or 'american' deal-by-deal).

    Returns:
        str: Markdown or JSON with per-tier LP/GP amounts, totals, LP MOIC,
            GP carry, and effective carry %. Schema (JSON):
            {"structure": str, "total_proceeds": float, "lp_distribution": float,
             "gp_distribution": float, "gp_carry": float, "lp_moic": float,
             "effective_carry_pct": float, "tiers": [{"name","lp","gp","total"}]}
    """
    try:
        terms = FundTerms(
            committed_capital=params.committed_capital,
            gp_commitment_pct=params.gp_commitment_pct,
            preferred_return=params.preferred_return,
            carry=params.carry,
        )
        if params.structure.lower() == "american":
            r = american_waterfall(terms, [params.total_proceeds],
                                   [params.committed_capital], [params.years])
        else:
            r = european_waterfall(terms, params.total_proceeds, years=params.years)

        data = {
            "structure": r.structure,
            "total_proceeds": round(r.total_proceeds, 2),
            "lp_distribution": round(r.lp_distribution, 2),
            "gp_distribution": round(r.gp_distribution, 2),
            "gp_carry": round(r.gp_carry, 2),
            "lp_moic": round(r.lp_moic, 4) if r.lp_moic else None,
            "effective_carry_pct": round(r.effective_carry_pct, 4) if r.effective_carry_pct else None,
            "tiers": [{"name": t.name, "lp": round(t.lp_amount, 2),
                       "gp": round(t.gp_amount, 2), "total": round(t.total, 2)} for t in r.tiers],
        }
        if params.response_format == ResponseFormat.JSON:
            return json.dumps(data, indent=2)

        lines = [f"# {r.structure} Waterfall", ""]
        lines.append(f"**Total proceeds:** {r.total_proceeds:,.1f}")
        lines.append("")
        lines.append("| Tier | LP | GP | Total |")
        lines.append("|------|----|----|-------|")
        for t in r.tiers:
            lines.append(f"| {t.name} | {t.lp_amount:,.1f} | {t.gp_amount:,.1f} | {t.total:,.1f} |")
        lines.append(f"| **TOTAL** | **{r.lp_distribution:,.1f}** | **{r.gp_distribution:,.1f}** | **{r.total_distributed:,.1f}** |")
        lines.append("")
        if r.lp_moic:
            lines.append(f"- **LP MOIC:** {r.lp_moic:.2f}x")
        lines.append(f"- **GP Carry:** {r.gp_carry:,.1f}")
        if r.effective_carry_pct:
            lines.append(f"- **Effective Carry:** {r.effective_carry_pct*100:.1f}%")
        return "\n".join(lines)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 2 — Fund Metrics (IRR / MOIC)
# ════════════════════════════════════════════════════════════════════════════

class MetricsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cashflows: List[float] = Field(..., description="Equal-period cash flows; negative=contribution, positive=distribution. e.g. [-100, 0, 0, 250]", min_length=2)
    residual_nav: float = Field(0.0, description="Unrealized residual NAV for TVPI/RVPI", ge=0)


@mcp.tool(
    name="dealdesk_fund_metrics",
    annotations={"title": "Compute Fund Metrics (IRR/MOIC)", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def dealdesk_fund_metrics(params: MetricsInput) -> str:
    """Compute fund performance metrics from a cash-flow stream.

    Args:
        params (MetricsInput): cashflows (negative=in, positive=out) and optional residual NAV.

    Returns:
        str: JSON with {"irr": float|null, "moic": float, "dpi": float,
            "tvpi": float, "contributions": float, "distributions": float}.
    """
    try:
        cfs = params.cashflows
        contributions = -sum(c for c in cfs if c < 0)
        distributions = sum(c for c in cfs if c > 0)
        data = {
            "irr": irr(cfs),
            "moic": moic(contributions, distributions, params.residual_nav),
            "dpi": dpi(contributions, distributions),
            "tvpi": tvpi(contributions, distributions, params.residual_nav),
            "contributions": round(contributions, 2),
            "distributions": round(distributions, 2),
        }
        return json.dumps(data, indent=2)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 3 — Secondary Pricing
# ════════════════════════════════════════════════════════════════════════════

class SecondaryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nav: float = Field(..., description="Net asset value of the LP stake", ge=0)
    bid_pct_nav: float = Field(..., description="Bid as fraction of NAV (0.90 = 10% discount, 1.05 = 5% premium)", ge=0)
    unfunded: float = Field(0.0, description="Remaining unfunded commitment", ge=0)


@mcp.tool(
    name="dealdesk_price_secondary",
    annotations={"title": "Price a Secondary LP Stake", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def dealdesk_price_secondary(params: SecondaryInput) -> str:
    """Price a secondary sale of an LP fund interest at a discount/premium to NAV.

    Args:
        params (SecondaryInput): nav, bid_pct_nav, optional unfunded commitment.

    Returns:
        str: JSON with {"price": float, "nav": float, "discount_premium": float,
            "is_discount": bool}.
    """
    try:
        q = price_secondary(params.nav, params.bid_pct_nav, unfunded=params.unfunded)
        return json.dumps({
            "price": q.price, "nav": q.nav,
            "discount_premium": q.discount_premium, "is_discount": q.is_discount,
        }, indent=2)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 4 — Continuation Vehicle
# ════════════════════════════════════════════════════════════════════════════

class CVInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_nav: float = Field(..., description="Current marked NAV of the asset", ge=0)
    purchase_price: float = Field(..., description="Price the continuation vehicle pays the old fund", ge=0)
    asset_cost_basis: float = Field(..., description="Original cost of the asset in the old fund", ge=0)
    rollover_pct: float = Field(..., description="Fraction of existing LPs rolling into the CV (0-1)", ge=0, le=1)
    carry: float = Field(0.20, description="Carried interest fraction", ge=0, lt=1)
    preferred_return: float = Field(0.08, description="Preferred return for crystallized-carry hurdle", ge=0, lt=1)
    holding_years: float = Field(5.0, description="Holding period in years", gt=0)


@mcp.tool(
    name="dealdesk_continuation_vehicle",
    annotations={"title": "Model GP-Led Continuation Vehicle", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def dealdesk_continuation_vehicle(params: CVInput) -> str:
    """Model a GP-led continuation vehicle: crystallized carry, cash-out vs roll.

    Args:
        params (CVInput): asset NAV, purchase price, cost basis, rollover %, carry, pref, years.

    Returns:
        str: JSON with {"crystallized_carry": float, "cashout_amount": float,
            "rollover_amount": float, "new_capital": float, "premium_to_nav": float}.
    """
    try:
        terms = FundTerms(committed_capital=max(params.asset_cost_basis, 1.0),
                          carry=params.carry, preferred_return=params.preferred_return)
        r = model_continuation_vehicle(
            params.asset_nav, params.purchase_price, params.asset_cost_basis,
            params.rollover_pct, terms, holding_years=params.holding_years)
        return json.dumps({
            "crystallized_carry": r.crystallized_carry,
            "cashout_amount": r.cashout_amount,
            "rollover_amount": r.rollover_amount,
            "new_capital": r.new_capital,
            "premium_to_nav": r.premium_to_nav,
        }, indent=2)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 5 — Exit Simulation (cap table)
# ════════════════════════════════════════════════════════════════════════════

class FounderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Founder/holder name", min_length=1)
    shares: float = Field(..., description="Number of shares", gt=0)


class RoundInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., description="Round name, e.g. 'Series A'", min_length=1)
    pre_money: float = Field(..., description="Pre-money valuation", gt=0)
    investment: float = Field(..., description="Amount invested in the round", gt=0)
    investor: str = Field(..., description="Investor name", min_length=1)
    liq_pref: float = Field(1.0, description="Liquidation preference multiple (1.0 = 1x)", ge=0)
    participating: bool = Field(False, description="Participating preferred?")


class ExitInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    founders: List[FounderInput] = Field(..., description="Founders and their share counts", min_length=1)
    esop: float = Field(0.0, description="ESOP pool shares", ge=0)
    rounds: List[RoundInput] = Field(default_factory=list, description="Priced funding rounds in order")
    exit_value: float = Field(..., description="Exit / sale value", ge=0)


@mcp.tool(
    name="dealdesk_simulate_exit",
    annotations={"title": "Simulate Cap-Table Exit Waterfall", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def dealdesk_simulate_exit(params: ExitInput) -> str:
    """Simulate an exit across a cap table with liquidation preferences.

    Builds the cap table from founders + ESOP + rounds, then distributes
    exit_value, handling each preferred holder's convert-vs-take-preference choice.

    Args:
        params (ExitInput): founders, esop, rounds, exit_value.

    Returns:
        str: JSON list of payouts: [{"name", "proceeds", "pct_of_exit",
            "multiple", "converted_to_common", "took_preference"}].
    """
    try:
        cap = CapTable()
        for f in params.founders:
            cap.add_founder(f.name, f.shares)
        if params.esop > 0:
            cap.add_esop(params.esop)
        for r in params.rounds:
            cap.apply_round(FundingRound(
                name=r.name, pre_money=r.pre_money, investment=r.investment,
                investor_name=r.investor, liq_pref_multiple=r.liq_pref,
                participating=r.participating))
        result = simulate_exit(cap, params.exit_value)
        payouts = [{
            "name": p.name, "proceeds": p.proceeds,
            "pct_of_exit": round(p.proceeds / params.exit_value, 4) if params.exit_value else 0,
            "multiple": round(p.multiple, 2) if p.multiple else None,
            "converted_to_common": p.converted_to_common,
            "took_preference": p.took_preference,
        } for p in result.payouts]
        return json.dumps({"exit_value": params.exit_value, "payouts": payouts}, indent=2)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 6 — Generate Investor Teaser
# ════════════════════════════════════════════════════════════════════════════

class TeaserFinancialInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    year: str = Field(..., description="Fiscal year label, e.g. 'FY2024'")
    revenue: float = Field(..., description="Revenue in ₹ Crore", ge=0)
    ebitda: Optional[float] = Field(None, description="EBITDA in ₹ Crore")
    pat: Optional[float] = Field(None, description="Profit after tax in ₹ Crore")


class TeaserInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    name: str = Field(..., description="Company name", min_length=1)
    sector: str = Field(..., description="Sector / industry", min_length=1)
    description: str = Field("", description="Short business description")
    headquarters: str = Field("", description="HQ city")
    transaction_type: str = Field("Growth Capital", description="e.g. 'M&A', 'Growth Capital', 'Secondary'")
    ask_amount_cr: Optional[float] = Field(None, description="Capital sought, ₹ Crore")
    highlights: List[str] = Field(default_factory=list, description="Investment highlight bullets")
    use_of_proceeds: List[str] = Field(default_factory=list, description="Use-of-proceeds bullets")
    financials: List[TeaserFinancialInput] = Field(default_factory=list, description="Annual financials")
    code_name: str = Field("", description="Code name for a blind teaser, e.g. 'Project Atlas'")
    anonymize: bool = Field(False, description="If true, produce a blind (anonymized) teaser")


@mcp.tool(
    name="dealdesk_generate_teaser",
    annotations={"title": "Generate Investor Teaser / CIM", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def dealdesk_generate_teaser(params: TeaserInput) -> str:
    """Generate an investor teaser (or anonymized blind teaser) from company info.

    Args:
        params (TeaserInput): company details, financials, highlights, ask, and
            anonymize flag.

    Returns:
        str: The full teaser rendered as markdown (overview, highlights, financial
            summary with auto-computed CAGR/margins, the ask, disclaimer).
    """
    try:
        company = CompanyInfo(
            name=params.name, sector=params.sector, description=params.description,
            headquarters=params.headquarters, transaction_type=params.transaction_type,
            ask_amount_cr=params.ask_amount_cr, highlights=params.highlights,
            use_of_proceeds=params.use_of_proceeds, code_name=params.code_name,
            financials=[FinancialYear(year=f.year, revenue=f.revenue,
                                      ebitda=f.ebitda, pat=f.pat) for f in params.financials],
        )
        teaser = generate_teaser(company, anonymize=params.anonymize)
        return render_markdown(teaser)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 7 — India Comparable Company Analysis (live NSE/BSE)
# ════════════════════════════════════════════════════════════════════════════

class CompsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    tickers: List[str] = Field(..., description="NSE/BSE tickers, e.g. ['INFY','TCS','WIPRO']", min_length=1, max_length=15)


@mcp.tool(
    name="dealdesk_india_comps",
    annotations={"title": "India Comparable Company Analysis", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def dealdesk_india_comps(params: CompsInput) -> str:
    """Build a comparable-company table for NSE/BSE tickers (live market data).

    Fetches EV/EBITDA, P/E, P/B, EV/Revenue multiples for each ticker and returns
    median/mean/high/low across the peer set. Requires network access.

    Args:
        params (CompsInput): list of NSE/BSE tickers.

    Returns:
        str: JSON with per-company multiples and summary statistics, or an Error.
    """
    if not _HAS_COMPS:
        return "Error: india-comps not installed. pip install git+https://github.com/bhupendra05/india-comps.git"
    try:
        multiples = []
        failed = []
        for t in params.tickers:
            try:
                _info, _fin, mult = _fetch_comps_company(t)
                multiples.append(mult)
            except Exception:
                failed.append(t)
        if not multiples:
            return f"Error: could not fetch any of: {', '.join(params.tickers)}"
        table = _build_comps(multiples)
        key_metrics = ["ev_ebitda", "pe_ratio", "pb_ratio", "ev_revenue"]
        out = {
            "companies": [{"symbol": m.symbol, "ev_ebitda": m.ev_ebitda,
                           "pe_ratio": m.pe_ratio, "pb_ratio": m.pb_ratio,
                           "ev_revenue": m.ev_revenue} for m in multiples],
            "summary": {metric: {"median": table.median(metric), "mean": table.mean(metric),
                                 "high": table.high(metric), "low": table.low(metric)}
                        for metric in key_metrics},
            "failed": failed,
        }
        return json.dumps(out, indent=2, default=str)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 8 — India DCF Valuation (live data, India-calibrated)
# ════════════════════════════════════════════════════════════════════════════

class IndiaDCFInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    ticker: str = Field(..., description="NSE/BSE ticker, e.g. 'INFY'", min_length=1)
    sector: str = Field("Default", description="Sector for beta/WC norms, e.g. 'IT Services', 'Banking', 'FMCG'")
    years: int = Field(5, description="Projection years", ge=3, le=10)
    terminal_growth: float = Field(0.055, description="Terminal growth (India default 5.5%)", ge=0, lt=0.10)


@mcp.tool(
    name="dealdesk_india_dcf",
    annotations={"title": "India-Calibrated DCF Valuation", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def dealdesk_india_dcf(params: IndiaDCFInput) -> str:
    """Run an India-calibrated DCF for an NSE/BSE company (live data).

    Uses G-Sec risk-free, India ERP, sector beta, 25.168% effective tax, ₹ Crore.
    Requires network access to fetch financials.

    Args:
        params (IndiaDCFInput): ticker, sector, projection years, terminal growth.

    Returns:
        str: JSON with WACC, implied prices (Gordon/exit/blended), enterprise value.
    """
    if not _HAS_DCF:
        return "Error: india-dcf not installed. pip install git+https://github.com/bhupendra05/india-dcf.git"
    try:
        company = _fetch_dcf(params.ticker, sector=params.sector)
        wacc = _calc_wacc(company, _WACCParams(beta=_SECTOR_BETA.get(params.sector, 1.0)))
        result = _run_dcf(company, _DCFAssump(projection_years=params.years,
                                              terminal_growth_rate=params.terminal_growth), wacc)
        return json.dumps({
            "company": result.company_name, "symbol": result.symbol,
            "wacc": round(result.wacc, 4), "tax_rate": round(result.tax_rate, 4),
            "enterprise_value_cr": round(result.ev_blended_cr, 1),
            "implied_price_gordon": round(result.implied_price_gordon, 2),
            "implied_price_exit": round(result.implied_price_exit, 2),
            "implied_price_blended": round(result.implied_price_blended, 2),
        }, indent=2)
    except Exception as e:
        return _err(e)


# ════════════════════════════════════════════════════════════════════════════
# Tool 9 — DRHP Red Flag Scan
# ════════════════════════════════════════════════════════════════════════════

class DRHPInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    pdf_path: str = Field(..., description="Absolute path to a DRHP PDF file", min_length=1)


@mcp.tool(
    name="dealdesk_drhp_analyze",
    annotations={"title": "DRHP Red Flag Scan", "readOnlyHint": True,
                 "destructiveHint": False, "idempotentHint": True, "openWorldHint": False},
)
async def dealdesk_drhp_analyze(params: DRHPInput) -> str:
    """Analyze a DRHP (IPO prospectus) PDF and surface a red-flag summary.

    Extracts company name, financials, objects of issue, RPTs, risk factors, and
    computes a red-flag score from the PDF.

    Args:
        params (DRHPInput): absolute path to a DRHP PDF.

    Returns:
        str: JSON summary with company, financials, red-flag score, or an Error.
    """
    if not _HAS_DRHP:
        return "Error: drhp-intel not installed. pip install git+https://github.com/bhupendra05/drhp-intel.git"
    try:
        text, pages = _read_pdf(params.pdf_path)
        summary = _drhp_analyze(text, pages)
        return json.dumps({
            "company_name": summary.company_name,
            "page_count": pages,
            "num_risk_factors": len(summary.risk_factors),
            "revenue_cagr": summary.revenue_cagr(),
            "red_flag_score": summary.red_flags.total if summary.red_flags else None,
        }, indent=2, default=str)
    except FileNotFoundError:
        return f"Error: file not found: {params.pdf_path}"
    except Exception as e:
        return _err(e)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
