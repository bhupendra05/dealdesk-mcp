# DealDesk MCP 🖥️

**Run PE/VC deal math inside Claude, ChatGPT, or any MCP client — just by chatting.**

DealDesk is an [MCP](https://modelcontextprotocol.io) server that exposes the
[CarryFlow](https://github.com/bhupendra05/carryflow),
[ExitSim](https://github.com/bhupendra05/exitsim), and
[TeaserGen](https://github.com/bhupendra05/teasergen) deal tools as chat-callable tools.

> *"Run a European waterfall on a ₹100 Cr fund returning ₹250 Cr at 8% pref, 20% carry"* —
> and the model just does it.

## Tools exposed

| Tool | What the LLM can do |
|------|---------------------|
| `dealdesk_run_waterfall` | European/American distribution waterfall, LP/GP split, carry |
| `dealdesk_fund_metrics` | IRR, MOIC, DPI, TVPI from a cash-flow stream |
| `dealdesk_price_secondary` | Price a secondary LP stake at discount/premium to NAV |
| `dealdesk_continuation_vehicle` | Model a GP-led continuation vehicle (carry, roll vs cash-out) |
| `dealdesk_simulate_exit` | Cap-table exit waterfall with liquidation preferences |
| `dealdesk_generate_teaser` | Generate an investor teaser or anonymized blind teaser |

## Install

```bash
pip install git+https://github.com/bhupendra05/dealdesk-mcp.git
```

## Connect to Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "dealdesk": {
      "command": "dealdesk-mcp"
    }
  }
}
```

Or run directly:

```bash
python -m dealdesk_mcp
```

## Example prompts

Once connected, ask your LLM:

- *"A fund committed ₹100 Cr and returned ₹300 Cr over 5 years. Run the waterfall at 8% pref and 20% carry — how much carry does the GP earn?"*
- *"Two founders own 6M and 4M shares. A Seed investor put in ₹2 Cr at ₹8 Cr pre with a 1x liq pref. If we exit at ₹4 Cr, who gets what?"*
- *"Generate a blind teaser for a robotics company doing ₹120 Cr revenue, 20% EBITDA margin, raising ₹150 Cr."*
- *"Price a secondary: NAV ₹50 Cr, bid at 88% of NAV."*

## Architecture

```
Claude / ChatGPT  ──MCP──▶  dealdesk-mcp  ──imports──▶  carryflow
                                                        exitsim
                                                        teasergen
```

All tools are **read-only** (no side effects), return structured JSON or markdown,
and validate inputs with Pydantic.

## License

MIT
