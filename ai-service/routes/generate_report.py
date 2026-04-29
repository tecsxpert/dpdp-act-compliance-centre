from flask import Blueprint

generate_report_bp = Blueprint('generate_report', __name__)

@generate_report_bp.route('/generate_report', methods=['POST'])
def generate_report():
    return {"message": "Generate Report endpoint coming soon"}, 200