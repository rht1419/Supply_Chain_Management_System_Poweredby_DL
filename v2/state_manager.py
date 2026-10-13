"""
v2/state_manager.py

Concurrent Real-Time Shared State Engine for DRAR Stores (v2).
Provides thread-safe atomic persistence across Storefront, SCM, and Warehouse Portals:
- Live Inventory with instant stock decrement and restocking
- Real-time customer transaction tracking
- Autonomous AI Replenishment Proposals
- Inbound Vendor Purchase Orders & Logistics Receiving
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = os.path.join("v2", "store_state.json")
INVENTORY_SOURCE = os.path.join("data", "inventory_status.json")

_STATE_LOCK = threading.Lock()


class StateManager:
    """
    Thread-safe state manager for DRAR Stores.
    Persists real-time synchronized state to v2/store_state.json.
    """

    def __init__(self, state_file: str = STATE_FILE):
        self.state_file = state_file
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Initializes state file from catalog and inventory data if not present."""
        with _STATE_LOCK:
            if not os.path.exists(self.state_file):
                logger.info(f"State file {self.state_file} does not exist. Initializing fresh state...")
                self._initialize_fresh_state()

    def _initialize_fresh_state(self):
        """Builds fresh baseline state merging 47 SKUs with rich catalog metadata."""
        from v2.catalog import RETAIL_CATALOG, get_product_metadata

        # Load inventory data from v1 pipeline if exists, else run inventory engine
        if not os.path.exists(INVENTORY_SOURCE):
            from src.inventory_optimizer import run_inventory_engine
            run_inventory_engine()

        with open(INVENTORY_SOURCE, "r", encoding="utf-8") as f:
            v1_inv = json.load(f)

        products = []
        for sku_row in v1_inv.get("skus", []):
            sku_id = int(sku_row["_ProductID"])
            meta = get_product_metadata(sku_id)

            # Map raw prices to realistic supermarket retail shelf prices
            raw_price = float(sku_row.get("unit_price", 25.0))
            raw_cost = float(sku_row.get("unit_cost", 18.0))

            # Scaled retail shelf price: normalize into typical retail consumer ranges ($4.99 to $89.99)
            if "Electronics" in meta["category"]:
                shelf_price = round(max(39.99, raw_price / 30.0), 2)
            else:
                shelf_price = round(max(4.49, raw_price / 120.0), 2)
            unit_cost = round(shelf_price * 0.65, 2)

            stock = int(sku_row["current_stock"])
            ss = int(sku_row["safety_stock"])
            rop = int(sku_row["reorder_point"])
            eoq = int(sku_row["eoq"])
            max_opt = rop + eoq

            status_code, status_label = self._classify_stock(stock, ss, rop, max_opt)

            products.append({
                "product_id": sku_id,
                "name": meta["name"],
                "category": meta["category"],
                "brand": meta["brand"],
                "icon": meta["icon"],
                "vendor": meta["vendor"],
                "shelf_price": shelf_price,
                "unit_cost": unit_cost,
                "current_stock": stock,
                "safety_stock": ss,
                "reorder_point": rop,
                "eoq": eoq,
                "max_optimal_stock": max_opt,
                "status_code": status_code,
                "status_label": status_label,
            })

        # Pre-seed realistic active AI proposals for items currently below ROP
        proposals = []
        for p in products:
            if p["current_stock"] < p["reorder_point"]:
                suggested_qty = max(p["eoq"], (p["reorder_point"] + p["eoq"]) - p["current_stock"])
                proposals.append({
                    "proposal_id": f"PROP-{datetime.now().strftime('%m%d')}-SKU{p['product_id']:03d}",
                    "product_id": p["product_id"],
                    "product_name": p["name"],
                    "category": p["category"],
                    "vendor": p["vendor"],
                    "current_stock": p["current_stock"],
                    "reorder_point": p["reorder_point"],
                    "safety_stock": p["safety_stock"],
                    "suggested_quantity": suggested_qty,
                    "unit_cost": p["unit_cost"],
                    "total_cost": round(suggested_qty * p["unit_cost"], 2),
                    "reason": f"Live inventory ({p['current_stock']} units) depleted below Reorder Point ({p['reorder_point']} units). Immediate EOQ replenishment required.",
                    "status": "PENDING",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })

        # Pre-seed a sample inbound purchase order for demo
        sample_po = {
            "po_id": f"PO-V2-{datetime.now().strftime('%Y%m%d')}-001",
            "created_at": (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "product_id": 1,
            "product_name": products[0]["name"],
            "vendor": products[0]["vendor"],
            "ordered_quantity": products[0]["eoq"],
            "unit_cost": products[0]["unit_cost"],
            "total_cost": round(products[0]["eoq"] * products[0]["unit_cost"], 2),
            "status": "Arrived at Warehouse",
            "approved_by": "Autonomous AI Agent",
            "eta": (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
            "history": [
                {"event": "Autonomous Restock PO Dispatched", "timestamp": (datetime.now() - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S")},
                {"event": "Carrier Dispatched from Supplier Depot", "timestamp": (datetime.now() - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S")},
                {"event": "Arrived at DRAR Regional Inbound Bay", "timestamp": (datetime.now() - timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M:%S")},
            ],
        }

        initial_state = {
            "store_name": "DRAR Stores Supercenter #104",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "products": products,
            "customer_orders": [],
            "purchase_orders": [sample_po],
            "ai_proposals": proposals,
        }

        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(initial_state, f, indent=2)
        logger.info(f"Fresh store state saved with {len(products)} SKUs and {len(proposals)} active AI proposals.")

    @staticmethod
    def _classify_stock(stock: int, ss: int, rop: int, max_opt: int) -> Tuple[str, str]:
        """Classifies stock health into the 4 standard operational tiers."""
        if stock < ss:
            return "CRITICAL_UNDERSTOCK", "🚨 Critical Stockout Risk (< SS)"
        elif stock < rop:
            return "REORDER_NEEDED", "⚠️ Reorder Needed (< ROP)"
        elif stock <= max_opt:
            return "OPTIMAL", "✅ Balanced / Optimal Stock"
        else:
            return "OVERSTOCK", "📦 Overstocking Surplus"

    def _read_state_locked(self) -> Dict[str, Any]:
        """Reads state JSON while inside an acquired lock."""
        if not os.path.exists(self.state_file):
            self._initialize_fresh_state()
        with open(self.state_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_state_locked(self, data: Dict[str, Any]):
        """Atomically writes state JSON while inside an acquired lock."""
        data["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tmp_file = f"{self.state_file}.tmp"
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_file, self.state_file)

    # ==========================================
    # PUBLIC API: Inventory Queries
    # ==========================================
    def get_inventory(self) -> List[Dict[str, Any]]:
        """Returns the live inventory list across all 47 products."""
        with _STATE_LOCK:
            data = self._read_state_locked()
            return data.get("products", [])

    def get_product(self, product_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves single product by SKU ID."""
        with _STATE_LOCK:
            data = self._read_state_locked()
            for p in data.get("products", []):
                if p["product_id"] == int(product_id):
                    return p
            return None

    # ==========================================
    # PUBLIC API: Customer & POS Transactions
    # ==========================================
    def place_customer_order(
        self,
        channel: str,
        customer_name: str,
        cart_items: List[Dict[str, Any]],
        discount_pct: float = 0.0,
        payment_method: str = "Credit Card",
    ) -> Dict[str, Any]:
        """
        Executes customer or cashier checkout:
        1. Immediately decrements live stock for each item in the cart.
        2. Recalculates inventory health status.
        3. Autonomously triggers an AI Restock Proposal if stock falls below ROP.
        4. Records transaction in customer_orders.
        """
        with _STATE_LOCK:
            data = self._read_state_locked()
            products_map = {p["product_id"]: p for p in data["products"]}

            order_items_detailed = []
            gross_total = 0.0
            newly_triggered_proposals = []

            for item in cart_items:
                pid = int(item["product_id"])
                qty = int(item["quantity"])
                if pid not in products_map:
                    continue

                prod = products_map[pid]
                # Decrement stock (cannot fall below 0)
                actual_deduct = min(prod["current_stock"], qty)
                prod["current_stock"] = max(0, prod["current_stock"] - qty)

                # Update health classification
                code, label = self._classify_stock(
                    prod["current_stock"],
                    prod["safety_stock"],
                    prod["reorder_point"],
                    prod["max_optimal_stock"],
                )
                prod["status_code"] = code
                prod["status_label"] = label

                subtotal = round(actual_deduct * prod["shelf_price"], 2)
                gross_total += subtotal

                order_items_detailed.append({
                    "product_id": pid,
                    "name": prod["name"],
                    "icon": prod["icon"],
                    "category": prod["category"],
                    "quantity": actual_deduct,
                    "unit_price": prod["shelf_price"],
                    "subtotal": subtotal,
                })

                # Check Autonomous Restock Trigger: Stock < ROP
                if prod["current_stock"] < prod["reorder_point"]:
                    # Check if a pending proposal already exists
                    has_pending = any(
                        prop["product_id"] == pid and prop["status"] == "PENDING"
                        for prop in data.get("ai_proposals", [])
                    )
                    if not has_pending:
                        suggested_qty = max(
                            prod["eoq"],
                            (prod["reorder_point"] + prod["eoq"]) - prod["current_stock"],
                        )
                        prop_obj = {
                            "proposal_id": f"PROP-{datetime.now().strftime('%m%d%H%M%S')}-SKU{pid:03d}",
                            "product_id": pid,
                            "product_name": prod["name"],
                            "category": prod["category"],
                            "vendor": prod["vendor"],
                            "current_stock": prod["current_stock"],
                            "reorder_point": prod["reorder_point"],
                            "safety_stock": prod["safety_stock"],
                            "suggested_quantity": suggested_qty,
                            "unit_cost": prod["unit_cost"],
                            "total_cost": round(suggested_qty * prod["unit_cost"], 2),
                            "reason": (
                                f"Live checkout by '{customer_name}' depleted stock to {prod['current_stock']} units, "
                                f"breaching Reorder Point ({prod['reorder_point']} units). Immediate vendor replenishment recommended."
                            ),
                            "status": "PENDING",
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        }
                        data.setdefault("ai_proposals", []).append(prop_obj)
                        newly_triggered_proposals.append(prop_obj)
                        logger.info(f"Autonomous AI replenishment proposal triggered for {prod['name']}!")

            discount_amt = round(gross_total * (discount_pct / 100.0), 2)
            net_total = round(gross_total - discount_amt, 2)
            order_id = f"ORD-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            order_record = {
                "order_id": order_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "channel": channel,
                "customer_name": customer_name,
                "items": order_items_detailed,
                "total_items": sum(it["quantity"] for it in order_items_detailed),
                "gross_subtotal": gross_total,
                "discount_pct": discount_pct,
                "discount_amount": discount_amt,
                "net_total": net_total,
                "payment_method": payment_method,
            }

            data.setdefault("customer_orders", []).insert(0, order_record)
            self._write_state_locked(data)

            return {
                "order": order_record,
                "new_proposals_count": len(newly_triggered_proposals),
                "new_proposals": newly_triggered_proposals,
            }

    def get_customer_orders(self) -> List[Dict[str, Any]]:
        """Returns recent customer/cashier orders."""
        with _STATE_LOCK:
            data = self._read_state_locked()
            return data.get("customer_orders", [])

    # ==========================================
    # PUBLIC API: AI Restock Proposals & Approval
    # ==========================================
    def get_ai_proposals(self, status: str = "ALL") -> List[Dict[str, Any]]:
        """Returns AI replenishment proposals, optionally filtered by status."""
        with _STATE_LOCK:
            data = self._read_state_locked()
            proposals = data.get("ai_proposals", [])
            if status.upper() == "ALL":
                return proposals
            return [p for p in proposals if p["status"].upper() == status.upper()]

    def approve_proposal(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """
        User / Manager clicks 'Approve & Dispatch PO to Vendor':
        1. Marks AI proposal as APPROVED.
        2. Automatically creates and dispatches a formal vendor Purchase Order (status: In Transit).
        """
        with _STATE_LOCK:
            data = self._read_state_locked()
            target_prop = None
            for p in data.get("ai_proposals", []):
                if p["proposal_id"] == proposal_id:
                    target_prop = p
                    break

            if not target_prop or target_prop["status"] != "PENDING":
                return None

            target_prop["status"] = "APPROVED"
            target_prop["approved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            po_id = f"PO-V2-{datetime.now().strftime('%Y%m%d')}-{len(data.get('purchase_orders', [])) + 1:03d}"
            new_po = {
                "po_id": po_id,
                "proposal_id": proposal_id,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "product_id": target_prop["product_id"],
                "product_name": target_prop["product_name"],
                "category": target_prop["category"],
                "vendor": target_prop["vendor"],
                "ordered_quantity": target_prop["suggested_quantity"],
                "unit_cost": target_prop["unit_cost"],
                "total_cost": target_prop["total_cost"],
                "status": "In Transit with Carrier",
                "approved_by": "User Manager Approval",
                "eta": (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d"),
                "history": [
                    {
                        "event": f"Restock Proposal Approved & Electronic PO Dispatched to {target_prop['vendor']}",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    {
                        "event": "Vendor EDI Acknowledged - Shipment In Transit via Express Freight",
                        "timestamp": (datetime.now() + timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
                    },
                ],
            }

            data.setdefault("purchase_orders", []).insert(0, new_po)
            self._write_state_locked(data)
            logger.info(f"Proposal {proposal_id} approved. Created PO {po_id} for {target_prop['product_name']}.")
            return new_po

    def decline_proposal(self, proposal_id: str) -> bool:
        """Declines / defers an AI replenishment proposal."""
        with _STATE_LOCK:
            data = self._read_state_locked()
            for p in data.get("ai_proposals", []):
                if p["proposal_id"] == proposal_id:
                    p["status"] = "DECLINED"
                    p["declined_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self._write_state_locked(data)
                    return True
            return False

    # ==========================================
    # PUBLIC API: Inbound Logistics & Receiving
    # ==========================================
    def get_purchase_orders(self) -> List[Dict[str, Any]]:
        """Returns all dispatched vendor Purchase Orders."""
        with _STATE_LOCK:
            data = self._read_state_locked()
            return data.get("purchase_orders", [])

    def receive_and_restock_po(self, po_id: str) -> Optional[Dict[str, Any]]:
        """
        Warehouse receiver clicks 'Receive Shipment & Restock Warehouse':
        1. Live on-hand inventory is immediately credited with ordered_quantity.
        2. Recalculates inventory health classification.
        3. Marks the Purchase Order as 'Fulfilled & Restocked'.
        """
        with _STATE_LOCK:
            data = self._read_state_locked()
            target_po = None
            for po in data.get("purchase_orders", []):
                if po["po_id"] == po_id:
                    target_po = po
                    break

            if not target_po or target_po["status"] == "Fulfilled & Restocked":
                return None

            pid = target_po["product_id"]
            qty = target_po["ordered_quantity"]

            # Credit inventory
            updated_prod = None
            for p in data["products"]:
                if p["product_id"] == pid:
                    p["current_stock"] += qty
                    code, label = self._classify_stock(
                        p["current_stock"],
                        p["safety_stock"],
                        p["reorder_point"],
                        p["max_optimal_stock"],
                    )
                    p["status_code"] = code
                    p["status_label"] = label
                    updated_prod = p
                    break

            # Update PO status
            target_po["status"] = "Fulfilled & Restocked"
            target_po["received_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            target_po.setdefault("history", []).append({
                "event": f"Shipment Docked & Received at Inbound Bay. Restocked +{qty} units to active shelf.",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

            self._write_state_locked(data)
            logger.info(f"PO {po_id} fulfilled! Restocked +{qty} units of Product #{pid}.")
            return {
                "po": target_po,
                "product": updated_prod,
            }

    def reset_demo_state(self):
        """Resets the state back to clean baseline."""
        with _STATE_LOCK:
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
            self._initialize_fresh_state()
            logger.info("Demo state reset completed.")
