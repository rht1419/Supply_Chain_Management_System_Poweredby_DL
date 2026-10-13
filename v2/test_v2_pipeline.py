"""
v2/test_v2_pipeline.py

Automated verification of the DRAR Stores v2 closed-loop retail lifecycle:
1. Initialize shared state engine
2. Customer order placement & stock decrement
3. Autonomous AI replenishment proposal generation (when Stock < ROP)
4. User PO approval and vendor dispatch
5. Inbound logistics arrival & warehouse receiving (stock replenishment)
6. Agentic Copilot response verification
"""

import os
import sys
import json

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from v2.state_manager import StateManager
from v2.nemotron_agent import NemotronStoreCopilot


def test_v2_closed_loop():
    print("=" * 60)
    print("RUNNING DRAR STORES V2 CLOSED-LOOP VERIFICATION")
    print("=" * 60)

    # 1. Initialize State Manager
    print("\n[Step 1] Initializing StateManager...")
    sm = StateManager()
    inv = sm.get_inventory()
    print(f"Loaded {len(inv)} active catalog products.")
    assert len(inv) == 47, f"Expected 47 products, got {len(inv)}"

    # Pick product #2 (Himalayan Basmati Rice)
    target_p = sm.get_product(2)
    print(f"Target Product: SKU #{target_p['product_id']} '{target_p['name']}'")
    initial_stock = target_p["current_stock"]
    rop = target_p["reorder_point"]
    print(f"Initial Stock: {initial_stock} units | ROP: {rop} units")

    # 2. Place Customer Order to deplete stock below ROP
    print("\n[Step 2] Simulating Customer In-Store Checkout...")
    order_qty = max(1, (initial_stock - rop) + 5)
    print(f"Placing order for {order_qty} units of SKU #2...")

    cart = [{"product_id": 2, "quantity": order_qty}]
    res = sm.place_customer_order(
        channel="In-Store POS (Cashier)",
        customer_name="Alex Mercer",
        cart_items=cart,
        discount_pct=5.0,
        payment_method="Apple Pay",
    )
    print(f"Order #{res['order']['order_id']} placed! Net Total: ${res['order']['net_total']:.2f}")

    updated_p = sm.get_product(2)
    print(f"Updated Stock: {updated_p['current_stock']} units (Status: {updated_p['status_label']})")
    assert updated_p["current_stock"] == max(0, initial_stock - order_qty)
    assert updated_p["current_stock"] < rop, "Stock should be below ROP"

    # 3. Verify Autonomous AI Replenishment Proposal
    print("\n[Step 3] Checking Autonomous AI Replenishment Proposals...")
    pending_props = sm.get_ai_proposals("PENDING")
    matching_prop = next((p for p in pending_props if p["product_id"] == 2), None)
    assert matching_prop is not None, "AI Proposal should have been triggered for SKU #2"
    print(f"Proposal Triggered: {matching_prop['proposal_id']}")
    print(f"Reason: {matching_prop['reason']}")
    print(f"Suggested Restock Quantity: {matching_prop['suggested_quantity']} units (${matching_prop['total_cost']:.2f})")

    # 4. User Approves Proposal & Dispatches PO
    print("\n[Step 4] User Approving Restock Proposal...")
    po = sm.approve_proposal(matching_prop["proposal_id"])
    assert po is not None, "Approval failed"
    print(f"Purchase Order Generated: {po['po_id']}")
    print(f"Vendor: {po['vendor']} | Status: {po['status']}")

    # 5. Warehouse Receiving & Restocking
    print("\n[Step 5] Warehouse Manager Receiving Inbound Shipment...")
    receive_res = sm.receive_and_restock_po(po["po_id"])
    assert receive_res is not None, "Receiving failed"
    restocked_p = receive_res["product"]
    print(f"Restocked SKU #2: Stock is now {restocked_p['current_stock']} units!")
    print(f"New Health Status: {restocked_p['status_label']}")
    assert restocked_p["current_stock"] > rop, "Stock should now be restored above ROP"

    # 6. Test Agentic AI Copilot Queries
    print("\n[Step 6] Testing Agentic AI Copilot...")
    copilot = NemotronStoreCopilot()
    q1 = copilot.query("Which items are critically understocked?")
    print("\n--- Copilot Stockout Response ---")
    print(q1[:300] + "...")

    q2 = copilot.query("Show 7-day demand forecast for SKU 2")
    print("\n--- Copilot Demand Forecast Response ---")
    print(q2[:300] + "...")

    print("\n" + "=" * 60)
    print("ALL DRAR STORES V2 CLOSED-LOOP TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    test_v2_closed_loop()
