"""
Self-Healing RL Recommendation Agent — A2A Orchestrator
Coordinates: Monitor → Diagnostics → Repair → Verification
"""

import os, sys, time, json, sqlite3

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from a2a.protocol import *
from agents.monitor_agent import MonitorAgent
from agents.diagnostics_agent import DiagnosticsAgent
from agents.repair_agent import RepairAgent
from agents.verification_agent import VerificationAgent
from mcp.tools import MCPTools
from config import *
from observability import traceable, status_line


class SelfHealingOrchestrator:
    def __init__(self):
        self.monitor = MonitorAgent()
        self.diagnostics = DiagnosticsAgent()
        self.repair = RepairAgent()
        self.verification = VerificationAgent()
        self.tools = MCPTools()
        self.healing_cycles = 0
        self.message_log = []

    @traceable(run_type="chain", name="self_healing_cycle")
    def run_healing_cycle(self, window: int = 50) -> bool:
        print(f"\n{'='*70}")
        print(f"🔄 SELF-HEALING CYCLE #{self.healing_cycles + 1}")
        print(f"{'='*70}")

        # Step 1: Monitor
        print(f"\n📍 Step 1: MONITOR AGENT")
        print(f"{'─'*40}")
        drift_alert = self.monitor.check_for_drift(window=window)

        if drift_alert is None:
            print(f"\n✅ No drift detected. System healthy.")
            return False

        self.message_log.append(drift_alert)
        self.healing_cycles += 1

        # Step 2: Diagnostics
        print(f"\n📍 Step 2: DIAGNOSTICS AGENT")
        print(f"{'─'*40}")
        repair_request = self.diagnostics.investigate(drift_alert)
        self.message_log.append(repair_request)

        # Step 3: Repair
        print(f"\n📍 Step 3: REPAIR AGENT")
        print(f"{'─'*40}")
        verify_request = self.repair.execute_repair(repair_request)
        self.message_log.append(verify_request)

        # Step 4: Verification
        print(f"\n📍 Step 4: VERIFICATION AGENT")
        print(f"{'─'*40}")
        result = self.verification.verify_repair(verify_request)
        self.message_log.append(result)

        if result.status == TaskStatus.COMPLETED:
            print(f"\n{'='*70}")
            print(f"✅ HEALING CYCLE COMPLETE — RL agent repaired")
            print(f"{'='*70}")
            self.monitor.handle_verification_result(result)
            return True
        elif result.status == TaskStatus.SENT_BACK:
            print(f"\n{'='*70}")
            print(f"↩️  HEALING INCOMPLETE — retrying")
            print(f"{'='*70}")
            retry = self.monitor.handle_verification_result(result)
            if retry:
                return self.run_healing_cycle(window=window)
        return True

    def print_summary(self):
        print(f"\n{'='*70}")
        print(f"📊 SELF-HEALING RL PIPELINE SUMMARY")
        print(f"{'='*70}")
        print(f"   Healing cycles: {self.healing_cycles}")
        print(f"   A2A messages: {len(self.message_log)}")

        actions = self.tools.get_agent_actions_summary()
        if actions:
            print(f"\n   Agent Actions:")
            for a in actions:
                print(f"     {a}")

        print(f"\n   Message Flow:")
        for msg in self.message_log:
            print_message(msg, prefix="     ")
        print(f"{'='*70}")


if __name__ == "__main__":
    orchestrator = SelfHealingOrchestrator()

    print("🎬 Self-Healing RL Recommendation Agent")
    print(status_line())
    print("Run these in separate terminals:")
    print("  1. python server.py")
    print("  2. python train_initial.py")
    print("  3. python drift_simulator.py 0.3")
    print("  4. python a2a/orchestrator.py")
    print()

    orchestrator.run_healing_cycle(window=50)
    orchestrator.print_summary()
