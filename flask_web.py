from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def park():
    return render_template('parking_qazaq.html')

if __name__ == '__main__':
    app.run(debug=True)
