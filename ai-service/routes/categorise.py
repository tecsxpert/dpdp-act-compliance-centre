from flask import Blueprint

categorise_bp = Blueprint('categorise', __name__)

@categorise_bp.route('/categorise', methods=['POST'])
def categorise():
    return {"message": "Categorise endpoint coming soon"}, 200