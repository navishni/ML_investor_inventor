from __future__ import annotations

import json
from functools import wraps
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for

from recommender import PlatformRecommender


BASE_DIR = Path(__file__).resolve().parent
PROFILE_PATH = BASE_DIR / "inventor_profiles.json"
USER_PATH = BASE_DIR / "users.json"

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


USERS = load_users()
PROFILES = load_profiles()


def save_users() -> None:
    save_json(USER_PATH, USERS)


def save_profiles() -> None:
    save_json(PROFILE_PATH, PROFILES)


def get_inventor_profile(idea_id: int) -> dict:
    return PROFILES.get(str(int(idea_id)), {})


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
    payload = {
        "username": user["username"],
        "role": user["role"],
        "entity_id": user["entity_id"],
        "display_name": user["display_name"],
    }
    if user["role"] == "investor":
        payload["profile"] = engine.get_investor(user["entity_id"])
        payload["recommendations"] = engine.recommend_for_investor(user["entity_id"])
    else:
        payload["profile"] = {
            **engine.get_inventor(user["entity_id"]),
            **get_inventor_profile(user["entity_id"]),
        }
        payload["matches"] = engine.recommend_for_inventor(user["entity_id"])
    return payload


def build_public_stats() -> dict:
    return {
        "demo_accounts": {
            "investor": {"username": "investor_1", "password": "demo123"},
            "inventor": {"username": "inventor_1", "password": "demo123"},
        },
        "model_comparison": engine.get_model_comparison(),
        "best_model": engine.best_model_name,
        "dataset": {
            "investors": int(len(engine.investors_df)),
            "inventors": int(len(engine.inventors_df)),
            "interactions": int(len(engine.history_df)),
            "positive_interactions": int((engine.history_df["interaction_score"] > 0).sum()),
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
