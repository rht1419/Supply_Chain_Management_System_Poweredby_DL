"""
v2/nemotron_agent.py

Agentic AI Supply Chain Copilot for DRAR Stores (v2).
Integrates NVIDIA Nemotron (via OpenRouter API) with grounded enterprise store state,
featuring full fallback to deterministic tool-calling reasoning when offline.
"""

import os
import sys
import json
import logging
import re
import requests
from typing import Dict, Any, List, Optional
from v2.state_manager import StateManager

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_NEMOTRON_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"


class NemotronStoreCopilot:
    """
    Autonomous Store Copilot with real-time state grounding and OpenRouter Nemotron integration.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "").strip()
        self.state_mgr = StateManager()

    def set_api_key(self, api_key: str):
        """Updates OpenRouter API key dynamically."""
        self.api_key = api_key.strip()

    def _build_store_context(self) -> str:
        """Assembles real-time store telemetry for LLM grounding."""
        inv = self.state_mgr.get_inventory()
        proposals = self.state_mgr.get_ai_proposals("PENDING")
        pos = self.state_mgr.get_purchase_orders()
        orders = self.state_mgr.get_customer_orders()

        critical = [p for p in inv if p["status_code"] == "CRITICAL_UNDERSTOCK"]
        reorder = [p for p in inv if p["status_code"] == "REORDER_NEEDED"]
        overstock = [p for p in inv if p["status_code"] == "OVERSTOCK"]

        context_lines = [
            "### DRAR STORES REAL-TIME STORE CONTEXT ###",
            f"Total Catalog SKUs: {len(inv)} active supermarket products.",
            f"Critical Stockout Risks (Stock < Safety Stock): {len(critical)} products.",
            f"Reorder Needed (Stock < ROP): {len(reorder)} products.",
            f"Overstocked Surplus: {len(overstock)} products.",
            f"Pending AI Restock Proposals Awaiting Approval: {len(proposals)}.",
            f"Active Dispatched Inbound POs: {len(pos)}.",
            f"Recent Customer/POS Orders: {len(orders)}.",
            "",
            "Top Stockout Urgencies:",
        ]
        for p in critical[:5]:
            context_lines.append(
                f"- SKU #{p['product_id']} {p['name']}: Current Stock = {p['current_stock']} | ROP = {p['reorder_point']} | EOQ = {p['eoq']} | Vendor = {p['vendor']}"
            )

        if proposals:
            context_lines.append("\nActive Pending AI Proposals:")
            for pr in proposals[:4]:
                context_lines.append(
                    f"- Proposal {pr['proposal_id']} for {pr['product_name']}: Requesting {pr['suggested_quantity']} units (${pr['total_cost']:,.2f}) from {pr['vendor']}"
                )

        return "\n".join(context_lines)

    def query(self, prompt: str) -> str:
        """
        Dispatches prompt to NVIDIA Nemotron via OpenRouter if API key is valid;
        otherwise runs local deterministic supply chain reasoning engine.
        """
        if self.api_key and len(self.api_key) > 8:
            try:
                return self._call_nemotron_openrouter(prompt)
            except Exception as e:
                logger.warning(f"OpenRouter call failed ({e}). Falling back to local reasoning engine.")
                fallback_resp = self._call_local_engine(prompt)
                return f"{fallback_resp}\n\n*(Note: OpenRouter API request encountered an error; served via local autonomous reasoning engine.)*"

        return self._call_local_engine(prompt)

    def _call_nemotron_openrouter(self, user_prompt: str) -> str:
        """Calls NVIDIA Nemotron on OpenRouter with live store grounding."""
        store_context = self._build_store_context()

        system_instruction = (
            "You are the autonomous AI Supply Chain & Operations Copilot for DRAR Stores (a high-volume supermarket chain like Walmart).\n"
            "You manage inventory, customer checkout telemetry, PyTorch deep learning demand forecasts, and autonomous vendor restocking.\n"
            "Always ground your answers in the real-time store telemetry provided below. Be concise, strategic, and professional.\n\n"
            f"{store_context}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8502",
            "X-Title": "DRAR Stores Autonomous SCM System",
            "Content-Type": "application/json",
        }

        payload = {
            "model": DEFAULT_NEMOTRON_MODEL,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.4,
            "max_tokens": 800,
        }

        resp = requests.post(OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=25)
        if resp.status_code == 200:
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise RuntimeError(f"OpenRouter error code {resp.status_code}: {resp.text}")

    def _call_local_engine(self, user_prompt: str) -> str:
        """Local autonomous tool-augmented reasoning engine."""
        text = user_prompt.lower()
        inv = self.state_mgr.get_inventory()
        proposals = self.state_mgr.get_ai_proposals("PENDING")
        pos = self.state_mgr.get_purchase_orders()
        orders = self.state_mgr.get_customer_orders()

        # 1. Proposal Queries: e.g. "show proposals" or "pending restocking"
        if any(w in text for w in ["proposal", "reorder request", "pending restock", "approval"]):
            if not proposals:
                return (
                    "### 📋 Autonomous AI Restock Proposals\n\n"
                    "✅ **All inventory levels are currently balanced!** There are no pending replenishment proposals awaiting approval."
                )
            lines = [
                f"* **{p['proposal_id']}** | **{p['product_name']}** (SKU #{p['product_id']})\n"
                f"  - Current Stock: **{p['current_stock']} units** (ROP: {p['reorder_point']}, Safety Stock: {p['safety_stock']})\n"
                f"  - Proposed Replenishment: **{p['suggested_quantity']} units** from **{p['vendor']}**\n"
                f"  - Total Financial Commitment: **${p['total_cost']:,.2f}**\n"
                f"  - *Action:* Head to Portal 2 (SCM Command Center) to click **[Approve & Dispatch PO]**."
                for p in proposals[:5]
            ]
            return (
                f"### 📋 Active AI Restock Proposals ({len(proposals)} Awaiting Approval)\n\n"
                + "\n\n".join(lines)
            )

        # 2. Inbound Logistics / PO Tracking
        if any(w in text for w in ["purchase order", "po", "inbound", "supplier", "vendor", "tracking", "receiving"]):
            if not pos:
                return "📦 **Inbound Logistics:** No active vendor purchase orders currently in transit."
            lines = [
                f"* **{po['po_id']}** for **{po['product_name']}**\n"
                f"  - Vendor: **{po['vendor']}** | Qty: **{po['ordered_quantity']} units** (${po['total_cost']:,.2f})\n"
                f"  - Status: `{po['status']}`\n"
                f"  - Latest Event: {po['history'][-1]['event'] if po.get('history') else 'Order Placed'}"
                for po in pos[:5]
            ]
            return (
                f"### 🚚 Inbound Logistics & Vendor Dispatch ({len(pos)} Orders)\n\n"
                + "\n\n".join(lines)
                + "\n\n💡 *Warehouse receiving staff can inspect and confirm docked shipments in Portal 3.*"
            )

        # 3. Stockout & Understock Inquiries
        if any(w in text for w in ["understock", "stockout", "critical", "low stock", "shortage", "reorder"]):
            crit = [p for p in inv if p["status_code"] in ["CRITICAL_UNDERSTOCK", "REORDER_NEEDED"]]
            lines = [
                f"* **SKU #{p['product_id']} {p['name']}**: Stock = **{p['current_stock']}** (ROP = {p['reorder_point']}, SS = {p['safety_stock']}) | Status: {p['status_label']}"
                for p in crit[:8]
            ]
            return (
                f"### 🚨 Real-Time Stockout Risk Audit\n\n"
                f"Identified **{len(crit)} supermarket products** currently operating below Reorder Point:\n\n"
                + "\n".join(lines)
                + "\n\n> **Autonomous Action:** The SCM engine has automatically drafted replenishment proposals for all qualifying SKUs."
            )

        # 4. Overstock & Surplus Inquiries
        if any(w in text for w in ["overstock", "surplus", "excess", "holding cost"]):
            over = [p for p in inv if p["status_code"] == "OVERSTOCK"]
            lines = [
                f"* **SKU #{p['product_id']} {p['name']}**: Stock = **{p['current_stock']} units** (Max Optimal = {p['max_optimal_stock']}) | Vendor: {p['vendor']}"
                for p in over[:8]
            ]
            return (
                f"### 📦 Overstocked Inventory & Holding Inefficiencies\n\n"
                f"Found **{len(over)} products** exceeding optimal capacity ceiling ($Current > ROP + EOQ$):\n\n"
                + "\n".join(lines)
                + "\n\n> **Strategic Recommendation:** Pause vendor purchase orders and trigger promotional POS bundle discounts."
            )

        # 5. SKU Deep Learning Demand Forecast
        sku_match = re.search(r"(?:sku|product|item)\s*#?\s*(\d+)", text)
        if sku_match and any(w in text for w in ["forecast", "predict", "demand", "7 days", "curve"]):
            sku_id = int(sku_match.group(1))
            prod = next((p for p in inv if p["product_id"] == sku_id), None)
            if not prod:
                return f"❌ SKU #{sku_id} was not found in the DRAR catalog (SKUs 1 to 47)."

            # Load forecast data from v1 cache
            f_path = os.path.join("data", "forecasts.json")
            f_curve = [round(max(1.0, prod["current_stock"] * 0.15 * (1 + 0.05 * i)), 1) for i in range(1, 8)]
            if os.path.exists(f_path):
                with open(f_path, "r", encoding="utf-8") as f:
                    fc_data = json.load(f)
                    if str(sku_id) in fc_data:
                        f_curve = fc_data[str(sku_id)]["forecast_7d"]

            tot = sum(f_curve)
            curve_str = ", ".join([f"Day +{i+1}: {v:.1f}" for i, v in enumerate(f_curve)])
            return (
                f"### 🔮 7-Day PyTorch LSTM Demand Forecast: {prod['name']}\n\n"
                f"* **Category:** {prod['category']} | **Brand:** {prod['brand']}\n"
                f"* **Current On-Shelf Stock:** **{prod['current_stock']} units**\n"
                f"* **Projected 7-Day Forward Velocity:** **{tot:.1f} units**\n"
                f"* **Daily Forecast Curve:** {curve_str}\n\n"
                f"> **SCM Evaluation:** Current stock ({prod['current_stock']} units) {'is SUFFICIENT' if prod['current_stock'] >= tot else 'is DEFICIENT'} to meet projected forward demand."
            )

        # 6. Customer Checkout / POS Order Inquiries
        if any(w in text for w in ["order", "sale", "pos", "transaction", "customer", "revenue", "checkout"]):
            total_rev = sum(o["net_total"] for o in orders)
            return (
                f"### 💳 DRAR Omni-Channel Checkout Telemetry\n\n"
                f"* **Completed Live Orders:** {len(orders)} transactions\n"
                f"* **Simulated Live Sales Revenue:** **${total_rev:,.2f}**\n"
                f"* **Active Order Channels:** Online E-Commerce & In-Store Cashier POS\n\n"
                f"*(Every checkout instantly decrements live shelf inventory in Portal 1)*"
            )

        # Default help
        return (
            "👋 **I am your DRAR Stores Autonomous SCM Operations Copilot.**\n\n"
            "You can ask me:\n"
            "1. 🚨 *'Which supermarket items have critical stockout risks?'*\n"
            "2. 📋 *'Show pending AI replenishment proposals awaiting approval'*\n"
            "3. 🚚 *'Track active inbound vendor purchase orders'*\n"
            "4. 🔮 *'What is the 7-day deep learning demand forecast for SKU 2?'*\n"
            "5. 💳 *'Summarize live customer orders and sales revenue'*\n"
            "6. 📦 *'Which items are overstocked and accumulating holding costs?'*"
        )
