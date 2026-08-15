# %%
import json

# %%


def process_workplan(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    
    project_name = data.get("project")
    total_planned_hours = data.get("totalPlannedHours")
    tasks = data.get("tasks", [])
    
    # 1. Separate phases and tasks
    phases = {t["id"]: t for t in tasks if t["taskType"] == "phase"}
    subtasks = [t for t in tasks if t["taskType"] == "task"]
    
    phase_summary = {}
    
    for phase_id, phase_info in phases.items():
        phase_name = phase_info["name"]
        phase_summary[phase_name] = {
            "duration_days": phase_info.get("durationDays"),
            "stated_hours": phase_info.get("plannedHours"),
            "calculated_role_hours": 0,
            "roles": [],
            "zero_hour_deliverables": [],
            "unmapped_items": []
        }
        
        # Filter child tasks for this phase
        child_tasks = [t for t in subtasks if t.get("parentId") == phase_id]
        
        for task in child_tasks:
            name = task["name"]
            hours = task.get("plannedHours", 0)
            assignment = task.get("assignment")
            
            # Categorize: Staffing Role vs Deliverable/Milestone Task
            if hours > 0:
                phase_summary[phase_name]["calculated_role_hours"] += hours
                phase_summary[phase_name]["roles"].append({
                    "role_name": name,
                    "assignment": assignment,
                    "hours": hours,
                    "work_code": task.get("workCode")
                })
            else:
                # 0-hour tasks indicate milestone or scope item needing definition
                phase_summary[phase_name]["zero_hour_deliverables"].append(name)

    # 2. Anomaly & Integrity Checks
    anomalies = []
    for p_name, p_data in phase_summary.items():
        if p_data["stated_hours"] != p_data["calculated_role_hours"]:
            anomalies.append(
                f"HOUR MISMATCH in {p_name}: Stated={p_data['stated_hours']} hrs, "
                f"Calculated Roles Sum={p_data['calculated_role_hours']} hrs"
            )
            
    # 3. Formulate Payload for LLM Synthesis
    llm_payload = {
        "project": project_name,
        "total_hours": total_planned_hours,
        "phase_breakdown": phase_summary,
        "qa_flags": anomalies
    }
    
    return llm_payload

# %%

# Execute processing
payload = process_workplan(r"C:\Users\Llubr\Downloads\Example_Media_Co_Workplan_v2_sanitized.json")
print(json.dumps(payload, indent=2))
# %%
