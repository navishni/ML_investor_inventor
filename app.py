from __future__ import annotations

import json
from functools import wraps
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for

from recommender import PlatformRecommender


BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "inventor_profiles.json"
USER_PATH = BASE_DIR / "users.json"
FEEDBACK_PATH = BASE_DIR / "recommendation_feedback.json"
CHAT_PATH = BASE_DIR / "chat_threads.json"
CHAT_READS_PATH = BASE_DIR / "chat_reads.json"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.secret_key = "matchtank-secret-key"

engine = PlatformRecommender()
engine.train()


def build_seed_users() -> dict[str, dict]:
    users = {}
    for row in engine.investors_df[["investor_id", "investor_name"]].to_dict(orient="records"):
        users[f"investor_{int(row['investor_id'])}"] = {
            "username": f"investor_{int(row['investor_id'])}",
            "password": "demo123",
            "role": "investor",
            "entity_id": int(row["investor_id"]),
            "display_name": row["investor_name"],
            "is_demo": True,
        }
    for row in engine.inventors_df[["idea_id", "idea_title"]].to_dict(orient="records"):
        users[f"inventor_{int(row['idea_id'])}"] = {
            "username": f"inventor_{int(row['idea_id'])}",
            "password": "demo123",
            "role": "inventor",
            "entity_id": int(row["idea_id"]),
            "display_name": f"Founder for {row['idea_title']}",
            "is_demo": True,
        }
    return users


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_users() -> dict[str, dict]:
    if USER_PATH.exists():
        return json.loads(USER_PATH.read_text(encoding="utf-8"))
    users = build_seed_users()
    save_json(USER_PATH, users)
    return users


def load_profiles() -> dict:
    if PROFILE_PATH.exists():
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profiles = {}
    for row in engine.inventors_df[["idea_id", "idea_title"]].to_dict(orient="records"):
        profiles[str(int(row["idea_id"]))] = {
            "idea_id": int(row["idea_id"]),
            "founder_name": f"Founder {int(row['idea_id'])}",
            "email": "",
            "linkedin": "",
            "achievements": "",
            "patents": "",
            "description": row["idea_title"],
        }
    save_json(PROFILE_PATH, profiles)
    return profiles


def load_json_store(path: Path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    save_json(path, default)
    return default


USERS = load_users()
PROFILES = load_profiles()
FEEDBACK = load_json_store(FEEDBACK_PATH, {})
CHATS = load_json_store(CHAT_PATH, {})
CHAT_READS = load_json_store(CHAT_READS_PATH, {})


def save_users() -> None:
    save_json(USER_PATH, USERS)


def save_profiles() -> None:
    save_json(PROFILE_PATH, PROFILES)


def save_feedback() -> None:
    save_json(FEEDBACK_PATH, FEEDBACK)


def save_chats() -> None:
    save_json(CHAT_PATH, CHATS)


def save_chat_reads() -> None:
    save_json(CHAT_READS_PATH, CHAT_READS)


def get_inventor_profile(idea_id: int) -> dict:
    return PROFILES.get(str(int(idea_id)), {})


def get_investor_feedback(investor_id: int) -> dict:
    return FEEDBACK.get(str(int(investor_id)), {})


def get_chat_sender_name(user: dict) -> str:
    if user["role"] == "inventor":
        profile = get_inventor_profile(user["entity_id"])
        founder_name = str(profile.get("founder_name", "")).strip()
        if founder_name:
            return founder_name
    return user["display_name"]


def _matches_query(values: list, query: str) -> bool:
    if not query:
        return True
    haystack = " ".join(str(value) for value in values if value not in (None, "")).lower()
    return query in haystack


def build_chat_contacts(user: dict, query: str = "", limit: int = 30) -> list[dict]:
    if not user:
        return []

    query = query.strip().lower()
    contacts: list[dict] = []

    if user["role"] == "investor":
        rows = engine.inventors_df.sort_values(["idea_title", "idea_id"], kind="stable").to_dict(orient="records")
        for row in rows:
            idea_id = int(row["idea_id"])
            profile = get_inventor_profile(idea_id)
            founder_name = str(profile.get("founder_name") or row.get("founder_name") or f"Founder {idea_id}").strip()
            values = [
                row.get("idea_title"),
                row.get("domain"),
                row.get("technology"),
                row.get("location"),
                row.get("risk_level"),
                row.get("idea_text"),
                founder_name,
                profile.get("achievements"),
                profile.get("patents"),
            ]
            if not _matches_query(values, query):
                continue
            contacts.append(
                {
                    "investor_id": int(user["entity_id"]),
                    "idea_id": idea_id,
                    "display_name": founder_name,
                    "counterparty_name": founder_name,
                    "subtitle": f"{row.get('idea_title', f'Idea {idea_id}')} | {row.get('domain', '')} | {row.get('technology', '')}",
                    "detail_line": f"{row.get('location', '')} | {row.get('risk_level', '')} risk",
                    "summary": _preview_text(row.get("idea_text", ""), 120),
                    "kind": "inventor",
                }
            )
            if len(contacts) >= limit:
                break
        return contacts

    rows = engine.investors_df.sort_values(["investor_name", "investor_id"], kind="stable").to_dict(orient="records")
    for row in rows:
        investor_id = int(row["investor_id"])
        values = [
            row.get("investor_name"),
            row.get("focus_domain"),
            row.get("industry_focus"),
            row.get("preferred_location"),
            row.get("preferred_risk_appetite"),
            row.get("available_funds"),
            row.get("company_investment"),
            row.get("past_investments"),
        ]
        if not _matches_query(values, query):
            continue
        contacts.append(
            {
                "investor_id": investor_id,
                "idea_id": int(user["entity_id"]),
                "display_name": row.get("investor_name", f"Investor {investor_id}"),
                "counterparty_name": row.get("investor_name", f"Investor {investor_id}"),
                "subtitle": f"{row.get('focus_domain', '')} | {row.get('industry_focus', '')}",
                "detail_line": f"{row.get('preferred_location', '')} | {row.get('preferred_risk_appetite', '')}",
                "summary": f"Available funds: {float(row.get('available_funds', 0)):,.0f} | Past investments: {int(row.get('past_investments', 0))}",
                "kind": "investor",
            }
        )
        if len(contacts) >= limit:
            break
    return contacts


def get_chat_thread(investor_id: int, idea_id: int) -> list[dict]:
    thread_id = f"{int(investor_id)}:{int(idea_id)}"
    return CHATS.get(thread_id, [])


def append_chat_message(investor_id: int, idea_id: int, sender_role: str, sender_name: str, message: str) -> list[dict]:
    thread_id = f"{int(investor_id)}:{int(idea_id)}"
    CHATS.setdefault(thread_id, [])
    CHATS[thread_id].append(
        {
            "sender_role": sender_role,
            "sender_name": sender_name,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_chats()
    return CHATS[thread_id]


def get_user_chat_reads(user: dict) -> dict:
    read_key = f"{user['role']}:{int(user['entity_id'])}"
    return CHAT_READS.setdefault(read_key, {})


def _thread_id_parts(thread_id: str) -> tuple[int, int]:
    investor_part, idea_part = thread_id.split(":", 1)
    return int(investor_part), int(idea_part)


def _preview_text(message: str, limit: int = 96) -> str:
    compact = " ".join(str(message).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def _thread_timestamp(messages: list[dict]) -> datetime:
    if not messages:
        return datetime.min.replace(tzinfo=timezone.utc)
    raw_timestamp = messages[-1].get("timestamp", "")
    try:
        parsed = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _message_timestamp(message: dict) -> datetime:
    raw_timestamp = message.get("timestamp", "")
    try:
        parsed = datetime.fromisoformat(raw_timestamp)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _count_unread_messages(user: dict, thread_id: str, messages: list[dict]) -> int:
    reads = get_user_chat_reads(user)
    last_read_raw = reads.get(thread_id, "")
    try:
        last_read = datetime.fromisoformat(last_read_raw) if last_read_raw else datetime.min.replace(tzinfo=timezone.utc)
    except ValueError:
        last_read = datetime.min.replace(tzinfo=timezone.utc)
    if last_read.tzinfo is None:
        last_read = last_read.replace(tzinfo=timezone.utc)

    unread_count = 0
    for message in messages:
        timestamp = _message_timestamp(message)
        if timestamp <= last_read:
            continue
        if message.get("sender_role") == user["role"]:
            continue
        unread_count += 1
    return unread_count


def mark_thread_read(user: dict, thread_id: str) -> None:
    reads = get_user_chat_reads(user)
    reads[thread_id] = datetime.now(timezone.utc).isoformat()
    save_chat_reads()


def build_chat_conversations(user: dict) -> list[dict]:
    if not user:
        return []

    role = user["role"]
    entity_id = int(user["entity_id"])
    conversations: list[dict] = []

    for thread_id, messages in CHATS.items():
        try:
            investor_id, idea_id = _thread_id_parts(thread_id)
        except ValueError:
            continue

        if role == "investor" and investor_id != entity_id:
            continue
        if role == "inventor" and idea_id != entity_id:
            continue

        investor = engine.get_investor(investor_id) or {}
        inventor = engine.get_inventor(idea_id) or {}
        profile = get_inventor_profile(idea_id)
        last_message = messages[-1] if messages else {}

        if role == "investor":
            label = inventor.get("idea_title") or profile.get("description") or f"Idea {idea_id}"
            counterparty_name = (
                profile.get("founder_name")
                or inventor.get("founder_name")
                or f"Founder for {label}"
            )
        else:
            label = investor.get("investor_name") or f"Investor {investor_id}"
            counterparty_name = investor.get("investor_name") or f"Investor {investor_id}"

        unread_count = _count_unread_messages(user, thread_id, messages)
        last_sender_role = last_message.get("sender_role", "")

        conversations.append(
            {
                "thread_id": thread_id,
                "investor_id": investor_id,
                "idea_id": idea_id,
                "label": label,
                "counterparty_name": counterparty_name,
                "last_sender": last_message.get("sender_name", ""),
                "last_sender_role": last_sender_role,
                "last_message_preview": _preview_text(last_message.get("message", "")),
                "message_count": len(messages),
                "unread_count": unread_count,
                "has_unread": unread_count > 0,
                "updated_at": _thread_timestamp(messages).isoformat(),
            }
        )

    conversations.sort(key=lambda row: row["updated_at"], reverse=True)
    return conversations


def json_error(message: str, status: int = 400):
    response = jsonify({"error": message})
    response.status_code = status
    return response


def get_current_user() -> dict | None:
    username = session.get("username")
    if not username:
        return None
    return USERS.get(username)


def login_required(role: str | None = None):
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            user = get_current_user()
            if not user:
                return redirect(url_for("login_page"))
            if role and user["role"] != role:
                return redirect(url_for("dashboard_router"))
            return view(*args, **kwargs)

        return wrapped

    return decorator


def build_session_payload(user: dict) -> dict:
    conversations = build_chat_conversations(user)
    payload = {
        "username": user["username"],
        "role": user["role"],
        "entity_id": user["entity_id"],
        "display_name": user["display_name"],
        "conversations": conversations,
        "chat_contacts": build_chat_contacts(user),
        "chat_unread_count": sum(int(thread.get("unread_count", 0)) for thread in conversations),
    }
    if user["role"] == "investor":
        payload["profile"] = engine.get_investor(user["entity_id"])
        payload["recommendations"] = engine.recommend_for_investor(user["entity_id"])
        payload["feedback"] = get_investor_feedback(user["entity_id"])
    else:
        payload["profile"] = {
            **engine.get_inventor(user["entity_id"]),
            **get_inventor_profile(user["entity_id"]),
        }
        payload["matches"] = engine.recommend_for_inventor(user["entity_id"])
    return payload


def build_public_stats() -> dict:
    like_count = sum(
        1
        for investor_feedback in FEEDBACK.values()
        for decision in investor_feedback.values()
        if decision == "like"
    )
    dislike_count = sum(
        1
        for investor_feedback in FEEDBACK.values()
        for decision in investor_feedback.values()
        if decision == "dislike"
    )
    message_count = sum(len(thread) for thread in CHATS.values())
    return {
        "demo_accounts": {
            "investor": {"username": "investor_1", "password": "demo123"},
            "inventor": {"username": "inventor_1", "password": "demo123"},
        },
        "model_comparison": engine.get_model_comparison(),
        "graph_payload": engine.get_graph_payload(),
        "market_insights": engine.get_market_insights(),
        "best_model": engine.best_model_name,
        "engagement": {
            "likes": like_count,
            "dislikes": dislike_count,
            "messages": message_count,
        },
        "dataset": {
            "investors": int(len(engine.investors_df)),
            "inventors": int(len(engine.inventors_df)),
            "interactions": int(len(engine.history_df)),
            "positive_interactions": int((engine.history_df["interaction_score"] >= 3).sum()),
        },
    }


@app.get("/")
def home():
    if get_current_user():
        return redirect(url_for("dashboard_router"))
    return redirect(url_for("login_page"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        user = USERS.get(username)
        if not user or user["password"] != password:
            flash("Invalid username or password.", "error")
        else:
            session["username"] = username
            return redirect(url_for("dashboard_router"))

    return render_template("login.html", bootstrap=build_public_stats())


@app.route("/signup", methods=["GET", "POST"])
def signup_page():
    if request.method == "POST":
        role = request.form.get("role", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        display_name = request.form.get("display_name", "").strip()
        selected_entity_id = request.form.get("entity_id", "").strip()

        if role not in {"investor", "inventor"}:
            flash("Choose a valid account type.", "error")
        elif not username or not password or not display_name or not selected_entity_id:
            flash("Fill in all required fields.", "error")
        elif username in USERS:
            flash("Username already exists. Choose another one.", "error")
        else:
            USERS[username] = {
                "username": username,
                "password": password,
                "role": role,
                "entity_id": int(selected_entity_id),
                "display_name": display_name,
                "is_demo": False,
            }
            save_users()
            session["username"] = username
            flash("Account created successfully.", "success")
            return redirect(url_for("dashboard_router"))

    investor_options = engine.investors_df[["investor_id", "investor_name"]].head(120).to_dict(orient="records")
    inventor_options = engine.inventors_df[["idea_id", "idea_title"]].head(120).to_dict(orient="records")
    return render_template(
        "signup.html",
        bootstrap=build_public_stats(),
        investor_options=investor_options,
        inventor_options=inventor_options,
    )


@app.get("/dashboard")
@login_required()
def dashboard_router():
    user = get_current_user()
    if user["role"] == "investor":
        return redirect(url_for("investor_dashboard"))
    return redirect(url_for("inventor_dashboard"))


@app.get("/analytics")
@login_required()
def analytics_page():
    return render_template(
        "analytics.html",
        bootstrap=build_public_stats(),
        user=get_current_user(),
    )


@app.get("/investor/dashboard")
@login_required("investor")
def investor_dashboard():
    user = get_current_user()
    return render_template(
        "investor_dashboard.html",
        user=user,
        payload=build_session_payload(user),
        bootstrap=build_public_stats(),
    )


@app.get("/inventor/dashboard")
@login_required("inventor")
def inventor_dashboard():
    user = get_current_user()
    return render_template(
        "inventor_dashboard.html",
        user=user,
        payload=build_session_payload(user),
        bootstrap=build_public_stats(),
    )


@app.get("/logout")
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.get("/api/bootstrap")
def bootstrap():
    return jsonify(build_public_stats())


@app.get("/api/analytics")
def analytics_api():
    return jsonify(
        {
            "graph_payload": engine.get_graph_payload(),
            "market_insights": engine.get_market_insights(),
            "model_comparison": engine.get_model_comparison(),
        }
    )


@app.post("/api/login")
def api_login():
    payload = request.get_json(force=True)
    username = payload.get("username", "").strip()
    password = payload.get("password", "").strip()
    user = USERS.get(username)
    if not user or user["password"] != password:
        return json_error("Invalid username or password.", 401)
    session["username"] = username
    return jsonify(
        {
            "user": build_session_payload(user),
            "model_comparison": engine.get_model_comparison(),
            "best_model": engine.best_model_name,
        }
    )


@app.get("/api/investor/<int:investor_id>/recommendations")
def investor_recommendations(investor_id: int):
    investor = engine.get_investor(investor_id)
    if not investor:
        return json_error("Investor not found.", 404)
    return jsonify(
        {
            "investor": investor,
            "recommendations": engine.recommend_for_investor(investor_id),
            "model_comparison": engine.get_model_comparison(),
            "best_model": engine.best_model_name,
        }
    )


@app.get("/api/inventor/<int:idea_id>/matches")
def inventor_matches(idea_id: int):
    inventor = engine.get_inventor(idea_id)
    if not inventor:
        return json_error("Inventor entry not found.", 404)
    return jsonify(
        {
            "inventor": {**inventor, **get_inventor_profile(idea_id)},
            "matches": engine.recommend_for_inventor(idea_id),
            "model_comparison": engine.get_model_comparison(),
            "best_model": engine.best_model_name,
        }
    )


@app.get("/api/inventor/<int:idea_id>/profile")
def inventor_profile(idea_id: int):
    inventor = engine.get_inventor(idea_id)
    if not inventor:
        return json_error("Inventor entry not found.", 404)
    return jsonify({**inventor, **get_inventor_profile(idea_id)})


@app.get("/api/inventor/<int:idea_id>/detail")
def inventor_detail(idea_id: int):
    inventor = engine.get_inventor(idea_id)
    if not inventor:
        return json_error("Inventor entry not found.", 404)
    return jsonify({**inventor, **get_inventor_profile(idea_id)})


@app.get("/api/investor/<int:investor_id>/detail")
def investor_detail(investor_id: int):
    investor = engine.get_investor(investor_id)
    if not investor:
        return json_error("Investor not found.", 404)
    return jsonify(investor)


@app.post("/api/inventor/<int:idea_id>/profile")
def save_inventor_profile_api(idea_id: int):
    inventor = engine.get_inventor(idea_id)
    if not inventor:
        return json_error("Inventor entry not found.", 404)

    payload = request.get_json(force=True)
    PROFILES[str(int(idea_id))] = {
        "idea_id": int(idea_id),
        "founder_name": payload.get("founder_name", ""),
        "email": payload.get("email", ""),
        "linkedin": payload.get("linkedin", ""),
        "achievements": payload.get("achievements", ""),
        "patents": payload.get("patents", ""),
        "description": payload.get("description", ""),
    }
    save_profiles()
    return jsonify({"message": "Inventor profile updated successfully."})


@app.post("/api/recommendation-feedback")
def recommendation_feedback():
    user = get_current_user()
    if not user or user["role"] != "investor":
        return json_error("Only investors can record feedback.", 403)

    payload = request.get_json(force=True)
    idea_id = int(payload.get("idea_id", 0))
    decision = payload.get("decision", "").strip().lower()
    if decision not in {"like", "dislike"}:
        return json_error("Decision must be like or dislike.")

    investor_feedback = FEEDBACK.setdefault(str(int(user["entity_id"])), {})
    investor_feedback[str(idea_id)] = decision
    save_feedback()
    return jsonify({"message": "Feedback saved.", "decision": decision, "idea_id": idea_id})


@app.get("/api/chat/<int:investor_id>/<int:idea_id>")
def get_chat(investor_id: int, idea_id: int):
    return jsonify({"messages": get_chat_thread(investor_id, idea_id)})


@app.post("/api/chat/<int:investor_id>/<int:idea_id>")
def post_chat(investor_id: int, idea_id: int):
    user = get_current_user()
    if not user:
        return json_error("Login required.", 401)

    payload = request.get_json(force=True)
    message = payload.get("message", "").strip()
    if not message:
        return json_error("Message cannot be empty.")

    sender_name = get_chat_sender_name(user)
    if user["role"] == "investor" and int(user["entity_id"]) != int(investor_id):
        return json_error("Investor mismatch.", 403)
    if user["role"] == "inventor" and int(user["entity_id"]) != int(idea_id):
        return json_error("Inventor mismatch.", 403)

    thread = append_chat_message(
        investor_id=investor_id,
        idea_id=idea_id,
        sender_role=user["role"],
        sender_name=sender_name,
        message=message,
    )
    mark_thread_read(user, f"{int(investor_id)}:{int(idea_id)}")
    return jsonify({"message": "Sent.", "messages": thread})


@app.post("/api/chat/<int:investor_id>/<int:idea_id>/read")
def mark_chat_read_api(investor_id: int, idea_id: int):
    user = get_current_user()
    if not user:
        return json_error("Login required.", 401)
    if user["role"] == "investor" and int(user["entity_id"]) != int(investor_id):
        return json_error("Investor mismatch.", 403)
    if user["role"] == "inventor" and int(user["entity_id"]) != int(idea_id):
        return json_error("Inventor mismatch.", 403)
    mark_thread_read(user, f"{int(investor_id)}:{int(idea_id)}")
    return jsonify({"message": "Thread marked as read."})


@app.get("/api/chat/inbox")
def chat_inbox_api():
    user = get_current_user()
    if not user:
        return json_error("Login required.", 401)
    conversations = build_chat_conversations(user)
    return jsonify(
        {
            "conversations": conversations,
            "unread_count": sum(int(thread.get("unread_count", 0)) for thread in conversations),
        }
    )


@app.get("/api/chat/contacts")
def chat_contacts_api():
    user = get_current_user()
    if not user:
        return json_error("Login required.", 401)
    query = request.args.get("q", "")
    limit = request.args.get("limit", 100, type=int)
    return jsonify({"contacts": build_chat_contacts(user, query=query, limit=limit)})


@app.post("/api/equity-calculator")
def equity_calculator():
    payload = request.get_json(force=True)
    try:
        valuation = float(payload.get("pre_money_valuation", 0))
        ask = float(payload.get("investment_amount", 0))
        founder_equity = float(payload.get("founder_equity_before", 100))
    except (TypeError, ValueError):
        return json_error("Enter valid numeric values.")

    if valuation <= 0 or ask <= 0:
        return json_error("Pre-money valuation and investment amount must be positive.")

    post_money = valuation + ask
    investor_equity = (ask / post_money) * 100
    founder_after = max(0.0, founder_equity - investor_equity)
    dilution = founder_equity - founder_after

    return jsonify(
        {
            "pre_money_valuation": valuation,
            "investment_amount": ask,
            "post_money_valuation": round(post_money, 2),
            "investor_equity_percent": round(investor_equity, 2),
            "founder_equity_after_percent": round(founder_after, 2),
            "dilution_percent": round(dilution, 2),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
