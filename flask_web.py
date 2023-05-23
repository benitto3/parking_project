from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def park():
    return render_template('parking_qazaq.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
    # type host-ip:5000 to see the webapp on other devices
