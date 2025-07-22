from flask import Flask, render_template, request, jsonify
from predictor import ReverseProteinPredictor

app = Flask(__name__)
predictor = ReverseProteinPredictor()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ss_sequence = request.form.get('ss_sequence')
        
        try:
            result = predictor.predict_structure(ss_sequence)
            return render_template('index.html', result=result, ss_sequence=ss_sequence)
        except Exception as e:
            return render_template('index.html', error=str(e), ss_sequence=ss_sequence)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
