from flask import Blueprint, request, jsonify
from src.pipeline.rag_engine import RAGEngine

api_bp = Blueprint("api", __name__, url_prefix="/api")

rag = RAGEngine()

@api_bp.route("/query", methods=["POST"])
def query_endpoint():
    data = request.get_json() or {}
    user_question = data.get("query")
    top_k = data.get("top_k", 4)

    if not user_question:
        return jsonify({"error": "Field 'query' is required."}), 400

    try:
        result_model = rag.query(user_question, top_k=top_k)
        return jsonify(result_model.model_dump()), 200
    except Exception as e:
        # Return exact error string to frontend
        status_code = 429 if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) else 500
        return jsonify({"error": str(e)}), status_code