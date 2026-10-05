from flask import Blueprint, request, jsonify
from pipeline.rag_engine import RAGEngine

api_bp = Blueprint("api", __name__, url_prefix="/api")

# Initialize engine instance
rag = RAGEngine()

@api_bp.route("/query", methods=["POST"])
def query_endpoint():
    data = request.get_json() or {}
    user_question = data.get("query")
    top_k = data.get("top_k", 4)

    if not user_question:
        return jsonify({"error": "Field 'query' is required."}), 400

    try:
        # Returns Answer Pydantic model
        result_model = rag.query(user_question, top_k=top_k)
        
        # Serialize Pydantic object to JSON-compatible dict
        return jsonify(result_model.model_dump()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500